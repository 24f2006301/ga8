import base64, hashlib, json, math, re, threading, unicodedata, zlib
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)
LOCK = threading.RLock()

# Stateful stores (persist for the lifetime of the Render instance).
BQM_RUNS = {}
FREEZES = {}
PIPELINES = {}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DEC = re.compile(r"^(0|[1-9][0-9]*)$")
URI_RE = re.compile(r"^gs://[^/]+/[^/]+$")
TS_RE = re.compile(r"^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2})(?:\\.(\\d{1,3}))?(Z|[+-]\\d{2}:\\d{2})$")

def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha_json(x):
    return sha_bytes(compact(x).encode("utf-8"))

def utf8key(x):
    return str(x).encode("utf-8")

def uniq_codes(codes):
    return sorted(set(codes), key=utf8key)

def finite_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def safe_int(x, positive=False):
    if not isinstance(x, int) or isinstance(x, bool):
        return False
    if x < (1 if positive else 0) or x > 9007199254740991:
        return False
    return True

def decimal_safe_string(x, positive=False):
    if not isinstance(x, str) or not DEC.fullmatch(x):
        return False
    try:
        n = int(x)
        return n >= (1 if positive else 0) and n <= 9007199254740991
    except Exception:
        return False

def parse_ts(s):
    if not isinstance(s, str):
        return None
    m = TS_RE.fullmatch(s)
    if not m:
        return None
    frac = (m.group(2) or "")
    frac = (frac + "000")[:3]
    tz = m.group(3)
    try:
        dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
        if tz == "Z":
            aware = dt.replace(tzinfo=timezone.utc)
        else:
            sign = 1 if tz[0] == "+" else -1
            hh, mm = map(int, tz[1:].split(":"))
            if hh > 14 or mm > 59 or (hh == 14 and mm != 0):
                return None
            off = timedelta(hours=hh, minutes=mm) * sign
            aware = dt.replace(tzinfo=timezone(off))
        return aware.replace(microsecond=int(frac) * 1000).astimezone(timezone.utc)
    except Exception:
        return None

def canon_ts(s):
    dt = parse_ts(s)
    return None if dt is None else dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond//1000:03d}Z"

def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return " ".join(s.split())

def json_obj(req):
    if not req.is_json:
        return None
    try:
        return req.get_json(force=False)
    except Exception:
        return None

def bad_input():
    return jsonify({"error":"INVALID_INPUT"}), 400

def crc32c(data):
    # Castagnoli CRC32C, pure Python table.
    poly = 0x82F63B78
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (poly if crc & 1 else 0)
    return f"{(crc ^ 0xFFFFFFFF) & 0xffffffff:08x}"

def words(s):
    # Unicode letter/number words, lowercased.
    s = unicodedata.normalize("NFKC", s).lower()
    out, cur = [], []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur: out.append("".join(cur))
    return set(out)

def jaccard(a,b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if a | b else 1.0

def compact_row(r):
    return compact({"id":r["id"],"entity":r["entity"],"eventTime":r["eventTime"],
                    "revision":r["revision"],"text":r["text"]})

# ---------------- 1. corpus ----------------
@app.post("/build-corpus")
def build_corpus():
    x = json_obj(request)
    if not isinstance(x, dict) or not isinstance(x.get("policy"), dict) or not isinstance(x.get("objects"), list):
        return bad_input()
    p = x["policy"]
    min_t, max_t = parse_ts(p.get("minTime")), parse_ts(p.get("maxTime"))
    threshold = p.get("contaminationThreshold")
    policy_ok = min_t is not None and max_t is not None and finite_num(threshold) and 0 <= float(threshold) <= 1
    objs_out, rows_out, lineage = [], [], []
    retained = []
    object_seen = []

    for o in x["objects"]:
        codes=[]; uri=o.get("uri"); content=o.get("content")
        if not isinstance(uri,str) or not URI_RE.fullmatch(uri): codes.append("URI_INVALID")
        g=o.get("generation"); fg=o.get("fetchedGeneration")
        if not decimal_safe_string(g): codes.append("GENERATION_INVALID")
        if not decimal_safe_string(fg): codes.append("GENERATION_INVALID")
        if isinstance(g,str) and isinstance(fg,str) and decimal_safe_string(g) and decimal_safe_string(fg) and g != fg:
            codes.append("GENERATION_MISMATCH")
        c=o.get("crc32c")
        crc_valid=isinstance(c,str) and re.fullmatch(r"[0-9a-f]{8}",c or "") is not None
        if not crc_valid: codes.append("CRC32C_INVALID")
        if isinstance(content,str) and crc_valid and crc32c(content.encode("utf-8")) != c:
            codes.append("CRC32C_MISMATCH")
        if o.get("schemaId") != "training-v1": codes.append("SCHEMA_INVALID")
        parsed=[]
        if not isinstance(content,str):
            codes.append("SCHEMA_INVALID")
        else:
            nonblank=False
            for line in content.splitlines():
                if not line.strip(): continue
                nonblank=True
                try: r=json.loads(line)
                except Exception:
                    codes.append("JSONL_INVALID"); parsed=[]; break
                if not isinstance(r,dict) or set(r.keys()) != {"id","entity","eventTime","revision","text"}:
                    codes.append("SCHEMA_INVALID"); continue
                if not all(isinstance(r.get(k),str) for k in ("id","entity","eventTime","text")) or not safe_int(r.get("revision")):
                    codes.append("SCHEMA_INVALID"); continue
                ct=canon_ts(r["eventTime"])
                if ct is None:
                    codes.append("SCHEMA_INVALID"); continue
                rr={"id":r["id"],"entity":norm_text(r["entity"]),"eventTime":ct,
                    "revision":r["revision"],"text":norm_text(r["text"])}
                parsed.append(rr)
            if not nonblank: codes.append("SCHEMA_INVALID")
        codes=uniq_codes(codes)
        if codes:
            objs_out.append({"uri":uri if isinstance(uri,str) else None,"reasonCodes":codes})
            continue
        object_seen.append((uri,g,c,o["schemaId"],parsed))

    # Deduplicate globally among otherwise valid rows.
    best={}
    losers=[]
    for uri,g,c,schema,rows in object_seen:
        for r in rows:
            key=(r["entity"],r["eventTime"],r["text"])
            if key not in best:
                best[key]=(r,uri,g,c,schema)
            else:
                old=best[key][0]
                winner = r if (r["revision"] > old["revision"] or
                                (r["revision"] == old["revision"] and utf8key(r["id"]) < utf8key(old["id"]))) else old
                loser = old if winner is r else r
                best[key]=(winner, uri if winner is r else best[key][1],
                           g if winner is r else best[key][2],
                           c if winner is r else best[key][3],
                           schema if winner is r else best[key][4])
                losers.append(loser)
    for r in losers:
        rows_out.append({"id":r["id"],"reasonCodes":["DUPLICATE"]})

    train=[]
    for r,uri,g,c,schema in best.values():
        codes=[]
        if not policy_ok:
            codes.append("POLICY_INVALID")
        else:
            dt=parse_ts(r["eventTime"])
            if dt < min_t or dt > max_t: codes.append("OUT_OF_WINDOW")
        if not codes:
            b=hashlib.sha256(r["entity"].encode("utf-8")).digest()[0] % 10
            split="train" if b<=5 else ("validation" if b<=7 else "test")
            r2=dict(r); r2["_split"]=split; r2["_words"]=words(r["entity"]+" "+r["text"])
            retained.append((r2,uri,g,c,schema))
            if split=="train": train.append(r2)
    train_words=[r["_words"] for r in train]
    for r,uri,g,c,schema in [z for z in retained if z[0]["_split"]!="train"]:
        if any(jaccard(r["_words"],tw) >= float(threshold) for tw in train_words):
            rows_out.append({"id":r["id"],"reasonCodes":["TRAIN_CONTAMINATION"]})
        else:
            pass

    splits={"train":[],"validation":[],"test":[]}
    lineage_map={}
    for r,uri,g,c,schema in retained:
        if any(q["id"]==r["id"] and "TRAIN_CONTAMINATION" in q["reasonCodes"] for q in rows_out):
            continue
        rr={k:r[k] for k in ("id","entity","eventTime","revision","text")}
        splits[r["_split"]].append(rr)
        lineage_map[uri]={"uri":uri,"generation":g,"crc32c":c,"schemaId":schema}
    for s in splits:
        splits[s].sort(key=lambda r:(utf8key(r["id"]),compact_row(r).encode("utf-8")))
    for s in splits:
        payload="".join(compact_row(r)+"\n" for r in splits[s]).encode("utf-8")
        splits[s] = splits[s]
    rows_out.sort(key=lambda r:(utf8key(r["id"]),compact(r).encode("utf-8")))
    objs_out.sort(key=lambda r:(utf8key("" if r["uri"] is None else r["uri"]),compact(r).encode("utf-8")))
    lineage=sorted(lineage_map.values(),key=lambda r:utf8key(r["uri"]))
    digests={}
    for s in ("train","validation","test"):
        digests[s]=sha_bytes(("".join(compact_row(r)+"\n" for r in splits[s])).encode("utf-8"))
    return jsonify({"splits":splits,"rejectedObjects":objs_out,"rejectedRows":rows_out,
                    "digests":digests,"lineage":lineage})

# ---------------- 2. bqml ----------------
def feature_selection(rows, forbidden):
    # dedupe by entity + UTC eventTime
    best={}
    for r in rows:
        et=canon_ts(r.get("eventTime"))
        if not isinstance(r,dict) or not isinstance(r.get("entity"),str) or et is None or not safe_int(r.get("version")) or not isinstance(r.get("id"),str):
            return None,None,None
        k=(r["entity"],et)
        if k not in best or r["version"]>best[k]["version"] or (r["version"]==best[k]["version"] and utf8key(r["id"])<utf8key(best[k]["id"])):
            best[k]=dict(r,_event=et)
    vals=list(best.values())
    if not vals: return None,None,None
    common=None
    for r in vals:
        f=r.get("features")
        if not isinstance(f,dict): return None,None,None
        names=set(f)
        common=names if common is None else common & names
    eligible=[]
    for name in common or set():
        if name in forbidden: continue
        ok=True
        for r in vals:
            v=r["features"][name]
            if not isinstance(v,dict) or "availableAt" not in v or parse_ts(v["availableAt"]) is None or parse_ts(v["availableAt"]) > parse_ts(r["predictionTime"]):
                ok=False; break
        if ok: eligible.append(name)
    eligible.sort(key=utf8key)
    tr=sorted([r["id"] for r in vals if r.get("split")=="TRAIN"],key=utf8key)
    ev=sorted([r["id"] for r in vals if r.get("split")=="EVAL"],key=utf8key)
    if any(r.get("split") not in ("TRAIN","EVAL") for r in vals): return None,None,None
    return tr,ev,eligible

@app.post("/bqml")
def bqml():
    x=json_obj(request)
    if not isinstance(x,dict) or x.get("phase") not in ("select","evaluate"):
        return bad_input()
    phase=x["phase"]; run=x.get("runId")
    if not isinstance(run,str) or not run or len(run)>128: return bad_input()
    if phase=="select":
        rows=x.get("rows"); trials=x.get("trials"); limit=x.get("numTrialsLimit")
        if not isinstance(rows,list) or not rows or not isinstance(trials,list) or not safe_int(limit,True):
            return bad_input()
        if len(trials)>limit: 
            out={"runId":run,"selectedTrialId":None,"trainRowIds":[],"evalRowIds":[],"featureNames":[],
                 "datasetDigest":None,"reasonCodes":["TRIAL_LIMIT_EXCEEDED"]}
        else:
            tids=set()
            ok=True
            for t in trials:
                if not isinstance(t,dict) or not safe_int(t.get("trialId")) or t["trialId"] in tids or t.get("status") not in ("SUCCEEDED","FAILED"):
                    ok=False; break
                tids.add(t["trialId"])
            forbidden=x.get("forbiddenFeatures",[])
            if not isinstance(forbidden,list) or not all(isinstance(z,str) for z in forbidden): ok=False
            tr,ev,fn=feature_selection(rows,forbidden)
            if not ok or tr is None:
                out={"runId":run,"selectedTrialId":None,"trainRowIds":[],"evalRowIds":[],"featureNames":[],
                     "datasetDigest":None,"reasonCodes":["INVALID_INPUT"]}
            else:
                good=[t for t in trials if t["status"]=="SUCCEEDED" and finite_num(t.get("evalMetric"))]
                if not good:
                    out={"runId":run,"selectedTrialId":None,"trainRowIds":tr,"evalRowIds":ev,"featureNames":fn,
                         "datasetDigest":sha_json({"trainRowIds":tr,"evalRowIds":ev,"featureNames":fn}),
                         "reasonCodes":["NO_SUCCESSFUL_TRIAL"]}
                else:
                    sel=sorted(good,key=lambda t:(-float(t["evalMetric"]),t["trialId"]))[0]["trialId"]
                    out={"runId":run,"selectedTrialId":sel,"trainRowIds":tr,"evalRowIds":ev,"featureNames":fn,
                         "datasetDigest":sha_json({"trainRowIds":tr,"evalRowIds":ev,"featureNames":fn}),"reasonCodes":[]}
        with LOCK:
            if run in BQM_RUNS:
                if compact(BQM_RUNS[run]["input"])==compact(x): return jsonify(BQM_RUNS[run]["output"])
                return jsonify({"error":"RUN_ID_CONFLICT"}),409
            BQM_RUNS[run]={"input":x,"output":out}
        return jsonify(out)
    # evaluate
    stored=BQM_RUNS.get(run)
    reason=[]; lineage_ok=True
    if not isinstance(x.get("selectedTrialId"),int) or x.get("selectedTrialId") is None or not HEX64.fullmatch(x.get("datasetDigest","")):
        reason.append("INVALID_INPUT"); lineage_ok=False
    if not stored or stored["output"].get("selectedTrialId") is None or x.get("selectedTrialId")!=stored["output"]["selectedTrialId"] or x.get("datasetDigest")!=stored["output"]["datasetDigest"]:
        reason.append("INVALID_LINEAGE"); lineage_ok=False
    metricFloor=x.get("metricFloor"); req=x.get("requiredSlices"); rows=x.get("rows")
    if not finite_num(metricFloor) or not 0<=float(metricFloor)<=1 or not isinstance(req,dict) or not isinstance(rows,list) or not isinstance(x.get("bytesProcessed"),int) or not isinstance(x.get("maxBytes"),int) or x["bytesProcessed"]<0 or x["maxBytes"]<0:
        reason.append("INVALID_INPUT"); lineage_ok=False
    test=None; slicepass=False
    validrows=True
    if isinstance(rows,list) and rows:
        for r in rows:
            if not isinstance(r,dict) or r.get("label") not in (0,1) or r.get("prediction") not in (0,1) or not isinstance(r.get("slice"),str) or not r["slice"]:
                validrows=False; break
        if validrows:
            test=round(sum(r["label"]==r["prediction"] for r in rows)/len(rows),12)
            for name,floor in req.items():
                sr=[r for r in rows if r["slice"]==name]
                if not sr: reason.append("MISSING_SLICE:"+name)
                elif not finite_num(floor) or not 0<=float(floor)<=1: reason.append("INVALID_INPUT")
                elif round(sum(r["label"]==r["prediction"] for r in sr)/len(sr),12)<float(floor): reason.append("SLICE_FLOOR:"+name)
            if test < float(metricFloor): reason.append("AGGREGATE_FLOOR")
        else:
            reason.append("INVALID_TEST_ROW")
    elif isinstance(rows,list):
        pass
    if not validrows: slicepass=False
    else:
        slicepass=not any(c.startswith("MISSING_SLICE:") or c.startswith("SLICE_FLOOR:") for c in reason) and lineage_ok and isinstance(rows,list) and bool(rows)
    if isinstance(x.get("bytesProcessed"),int) and isinstance(x.get("maxBytes"),int) and x["bytesProcessed"]>x["maxBytes"]: reason.append("BYTE_LIMIT")
    out={"runId":run,"selectedTrialId":x.get("selectedTrialId"),"datasetDigest":x.get("datasetDigest"),
         "testMetric":test,"criticalSlicePass":bool(slicepass),"decision":"admit" if not reason else "reject",
         "bytesProcessed":x.get("bytesProcessed"),"reasonCodes":uniq_codes(reason)}
    return jsonify(out)

# ---------------- 3. promote ----------------
@app.post("/promote")
def promote():
    x=json_obj(request)
    if not isinstance(x,dict) or not isinstance(x.get("policy"),dict) or not isinstance(x.get("versions"),list) or not isinstance(x.get("championVersion"),str):
        return bad_input()
    p=x["policy"]; asof=parse_ts(x.get("asOf")); vs=x["versions"]
    base={"action":"block","championVersion":x["championVersion"],"selectedVersion":None,"eligibleVersions":[],
          "failedGates":{},"aliasMutation":None,"evidence":None}
    if asof is None: 
        for v in vs: base["failedGates"][v.get("version","")]=["INVALID_TIMESTAMP"]
        return jsonify(base)
    seen=set(); invalid_dups=set()
    for v in vs:
        if not isinstance(v,dict) or not decimal_safe_string(v.get("version"),True) or v["version"] in seen:
            invalid_dups.add(v.get("version","")); seen.add(v.get("version",""))
        else: seen.add(v["version"])
    req=p.get("requiredSlices")
    policy_ok=(all(isinstance(p.get(k),str) and p.get(k) for k in ("datasetDigest","schemaDigest")) and
               safe_int(p.get("maxAgeSeconds")) and finite_num(p.get("accuracyFloor")) and 0<=p["accuracyFloor"]<=1 and
               finite_num(p.get("maxLatencyMs")) and p["maxLatencyMs"]>=0 and safe_int(p.get("maxSizeBytes")) and
               finite_num(p.get("minImprovement")) and 0<=p["minImprovement"]<=1 and isinstance(req,dict))
    eligible=[]
    for v in vs:
        name=v.get("version",""); codes=[]
        if name in invalid_dups: codes.append("DUPLICATE_VERSION" if name in seen else "INVALID_VERSION")
        if not decimal_safe_string(name,True): codes.append("INVALID_VERSION")
        if not policy_ok: codes.append("INVALID_POLICY")
        e=v.get("evaluation")
        if not isinstance(e,dict): codes.append("MISSING_EVALUATION")
        else:
            ct=parse_ts(e.get("createdAt"))
            if ct is None: codes.append("INVALID_TIMESTAMP")
            else:
                if ct>asof: codes.append("FUTURE_EVALUATION")
                if ct<asof-timedelta(seconds=p.get("maxAgeSeconds",0)): codes.append("STALE_EVALUATION")
            vals=(e.get("accuracy"),e.get("latencyMs"),e.get("sizeBytes"))
            if not all(finite_num(vals[0:1][0]) if False else True for _ in [0]): pass
            if not finite_num(e.get("accuracy")) or not finite_num(e.get("latencyMs")) or not isinstance(e.get("sizeBytes"),int): codes.append("NON_FINITE")
            if finite_num(e.get("accuracy")) and not 0<=e["accuracy"]<=1: codes.append("METRIC_RANGE")
            if finite_num(e.get("latencyMs")) and e["latencyMs"]<0: codes.append("METRIC_RANGE")
            if isinstance(e.get("sizeBytes"),int) and (e["sizeBytes"]<0 or e["sizeBytes"]>9007199254740991): codes.append("METRIC_RANGE")
            if v.get("artifactDigest") != e.get("artifactDigest"): codes.append("ARTIFACT_MISMATCH")
            if e.get("datasetDigest") != p.get("datasetDigest"): codes.append("DATASET_MISMATCH")
            if e.get("schemaDigest") != p.get("schemaDigest"): codes.append("SCHEMA_MISMATCH")
            if finite_num(e.get("accuracy")) and e["accuracy"]<p.get("accuracyFloor",0): codes.append("ACCURACY_FLOOR")
            if finite_num(e.get("latencyMs")) and e["latencyMs"]>p.get("maxLatencyMs",float("inf")): codes.append("LATENCY_LIMIT")
            if isinstance(e.get("sizeBytes"),int) and e["sizeBytes"]>p.get("maxSizeBytes",9007199254740991): codes.append("SIZE_LIMIT")
            sl=e.get("slices")
            if not isinstance(sl,dict): sl={}
            for n,f in req.items():
                if n not in sl: codes.append("MISSING_SLICE:"+n)
                elif not finite_num(sl[n]) or not 0<=sl[n]<=1: codes.append("SLICE_RANGE:"+n)
                elif sl[n]<f: codes.append("SLICE_FLOOR:"+n)
        codes=uniq_codes(codes)
        if codes: base["failedGates"][name]=codes
        else: eligible.append(v)
    eligible.sort(key=lambda v:(-v["evaluation"]["accuracy"],v["evaluation"]["latencyMs"],v["evaluation"]["sizeBytes"],int(v["version"])))
    base["eligibleVersions"]=[v["version"] for v in eligible]
    champ=next((v for v in vs if v.get("version")==x["championVersion"]),None)
    champ_ok=champ is not None and x["championVersion"] in [v["version"] for v in eligible]
    if not champ_ok:
        base["action"]="block"; return jsonify(base)
    if not eligible: return jsonify(base)
    challenger=eligible[0]
    if challenger["version"]==champ["version"] or round(challenger["evaluation"]["accuracy"]-champ["evaluation"]["accuracy"],12) < p["minImprovement"]:
        base["action"]="retain"; base["selectedVersion"]=champ["version"]; base["evidence"]=champ["evaluation"]; return jsonify(base)
    base["action"]="promote"; base["selectedVersion"]=challenger["version"]; base["aliasMutation"]={"alias":"champion","version":challenger["version"]}; base["evidence"]=challenger["evaluation"]
    return jsonify(base)

# ---------------- 4. adapt ----------------
@app.post("/adapt")
def adapt():
    x=json_obj(request)
    if not isinstance(x,dict) or x.get("operation") not in ("choose","repair"): return bad_input()
    if x["operation"]=="choose":
        p=x.get("policy"); cs=x.get("candidates")
        names=["prompt_only","retrieval","lora","qlora"]
        if not isinstance(p,dict) or not isinstance(cs,list) or len(cs)!=4: return bad_input()
        by={c.get("name"):c for c in cs if isinstance(c,dict)}
        eligible=[]; costs={}; reasons={}
        for n in names:
            c=by.get(n); rc=[]
            if c is None: rc=["INVALID_INPUT"]; costs[n]=None; reasons[n]=rc; continue
            if c.get("available") is not True: rc.append("UNAVAILABLE")
            if not finite_num(c.get("quality")) or not 0<=c.get("quality",0)<=1 or not finite_num(p.get("minQuality")) or not 0<=p.get("minQuality",0)<=1 or c.get("quality",0)<p.get("minQuality",0): rc.append("QUALITY_FLOOR")
            if p.get("freshnessRequired") is True and c.get("freshness") is not True: rc.append("FRESHNESS_REQUIRED")
            if not finite_num(c.get("latencyMs")) or not finite_num(p.get("maxLatencyMs")) or c.get("latencyMs")>p.get("maxLatencyMs"): rc.append("LATENCY_LIMIT")
            if not finite_num(c.get("memoryMb")) or not finite_num(p.get("maxMemoryMb")) or c.get("memoryMb")>p.get("maxMemoryMb"): rc.append("MEMORY_LIMIT")
            if not safe_int(c.get("labeledExamples")) or not safe_int(p.get("maxLabeledExamples")) or c.get("labeledExamples")>p.get("maxLabeledExamples"): rc.append("DATA_LIMIT")
            if not finite_num(c.get("oneTimeCost")) or not finite_num(c.get("recurringCost")) or not safe_int(p.get("horizonRequests")) or not finite_num(p.get("maxTotalCost")): rc.append("COST_LIMIT"); total=None
            else:
                total=round(c["oneTimeCost"]+p["horizonRequests"]*c["recurringCost"],12)
                if c["oneTimeCost"]<0 or c["recurringCost"]<0 or total>p["maxTotalCost"]: rc.append("COST_LIMIT")
            costs[n]=total; reasons[n]=uniq_codes(rc)
            if not rc: eligible.append(n)
        return jsonify({"selected":eligible[0] if eligible else None,"eligible":eligible,"totalCosts":costs,"reasonCodes":reasons})
    # repair
    toks=x.get("tokens"); params=x.get("parameters"); allowed=x.get("allowedTargets")
    rc=[]; labels=[]
    validtok=isinstance(toks,list) and bool(toks)
    if validtok:
        for t in toks:
            if not isinstance(t,dict) or not safe_int(t.get("id")) or t.get("role") not in ("system","user","assistant") or not isinstance(t.get("padding"),bool) or not isinstance(t.get("text"),str):
                validtok=False; break
    labels=[t["id"] if validtok and t["role"]=="assistant" and not t["padding"] else -100 for t in toks] if isinstance(toks,list) else []
    if not validtok: rc.append("INVALID_TOKEN"); labels=[-100]*(len(toks) if isinstance(toks,list) else 0)
    if x.get("templateApplications")!=1: rc.append("CHAT_TEMPLATE_COUNT")
    if not isinstance(params,list) or not isinstance(allowed,list) or not allowed or len(set(allowed))!=len(allowed) or not all(isinstance(a,str) and a for a in allowed): rc.append("INVALID_PARAMETER")
    train=[]
    if isinstance(params,list):
        seen=set()
        for p in params:
            if not isinstance(p,dict) or not isinstance(p.get("name"),str) or p["name"] in seen or not safe_int(p.get("numel"),True) or not isinstance(p.get("target"),str):
                rc.append("INVALID_PARAMETER"); continue
            seen.add(p["name"])
            if p["target"] in allowed and (p["name"].endswith(".lora_A.weight") or p["name"].endswith(".lora_B.weight")): train.append(p)
    if not train: rc.append("INVALID_PARAMETER")
    if x.get("inferenceMode") is not False: rc.append("INFERENCE_MODE")
    if x.get("dropoutActiveDuringEval") is not False: rc.append("EVAL_DROPOUT_ACTIVE")
    tr=x.get("trainRowIds"); ev=x.get("evalRowIds")
    if not isinstance(tr,list) or not isinstance(ev,list) or not tr or not ev or len(set(tr))!=len(tr) or len(set(ev))!=len(ev) or set(tr)&set(ev) or not all(isinstance(i,str) and i for i in tr+ev): rc.append("EVAL_LEAKAGE")
    af=x.get("artifactFiles")
    if not isinstance(af,list) or sorted(af)!=["adapter_config.json","adapter_model.safetensors"] or len(af)!=2: rc.append("ADAPTER_FILE_SET")
    if x.get("baseRevision") is not None and not HEX40.fullmatch(x.get("baseRevision","")): rc.append("MUTABLE_BASE_REVISION")
    for k in ("datasetDigest","codeDigest","configDigest"):
        if not HEX64.fullmatch(x.get(k,"")): rc.append("LINEAGE_MISMATCH")
    if not all(isinstance(x.get(k),int) and x[k]>0 and x[k]<=9007199254740991 for k in ("microBatch","gradientAccumulation","replicas","expectedEffectiveBatch")) or x["microBatch"]*x["gradientAccumulation"]*x["replicas"]!=x["expectedEffectiveBatch"]: rc.append("EFFECTIVE_BATCH_MISMATCH")
    cp=x.get("checkpoint")
    if not isinstance(cp,dict) or not all(k in cp for k in ("model","optimizer","scheduler","step","rng","dataPosition")): rc.append("INCOMPLETE_CHECKPOINT")
    uw,rw,tol=x.get("uninterruptedWeights"),x.get("resumedWeights"),x.get("resumeTolerance")
    resume=True
    if not isinstance(uw,list) or not isinstance(rw,list) or not uw or len(uw)!=len(rw) or not finite_num(tol) or tol<0 or not all(finite_num(a) for a in uw+rw): resume=False
    elif any(abs(a-b)>tol for a,b in zip(uw,rw)): resume=False
    if not resume: rc.append("RESUME_DIVERGENCE")
    # Full-model artifact is signalled when arbitrary full-model files are supplied; this request format has adapterFiles only.
    train.sort(key=lambda p:utf8key(p["name"]))
    count=sum(p["numel"] for p in train)
    return jsonify({"labels":labels,"templatePass":x.get("templateApplications")==1,"trainableParams":[p["name"] for p in train],
                    "trainableCount":count,"peftConfigPass":not ("INVALID_PARAMETER" in rc),
                    "adapterFiles":sorted(af,key=utf8key) if isinstance(af,list) else [],
                    "checkpointComplete":isinstance(cp,dict) and all(k in cp for k in ("model","optimizer","scheduler","step","rng","dataPosition")),
                    "lineagePass":not any(c in rc for c in ("MUTABLE_BASE_REVISION","LINEAGE_MISMATCH")),
                    "evalIsolated":"EVAL_LEAKAGE" not in rc,"evaluationDeterministic":"EVAL_DROPOUT_ACTIVE" not in rc,
                    "resumePass":resume,"reasonCodes":uniq_codes(rc)})

# ---------------- 5. quantize ----------------
@app.post("/quantize")
def quantize():
    x=json_obj(request)
    if not isinstance(x,dict) or x.get("phase") not in ("freeze","select"): return bad_input()
    if x["phase"]=="freeze":
        fid=x.get("freezeId"); cs=x.get("candidates")
        if not isinstance(fid,str) or not fid or len(fid)>128 or not isinstance(cs,list): return bad_input()
        allowed=x.get("allowedUnsupportedReasons",[])
        if not isinstance(allowed,list) or len(set(allowed))!=len(allowed) or not all(isinstance(a,str) and a for a in allowed): return bad_input()
        outc=[]
        for c in cs:
            rc=[]; inv=[]; total=None; pkg=None
            if not isinstance(c,dict) or not isinstance(c.get("name"),str) or not c["name"] or not isinstance(c.get("files"),dict) or not c["files"] or any(not isinstance(k,str) or not isinstance(v,str) for k,v in c["files"].items()):
                rc.append("INVALID_INPUT")
            else:
                for n,v in c["files"].items():
                    inv.append({"name":n,"bytes":len(v.encode("utf-8")),"sha256":sha_bytes(v.encode("utf-8"))})
                inv.sort(key=lambda z:utf8key(z["name"]))
                total=sum(i["bytes"] for i in inv); pkg=sha_json(inv)
                ur=c.get("unsupportedReason")
                if ur is not None:
                    if not isinstance(ur,str) or ur not in allowed: rc.append("UNALLOWED_UNSUPPORTED_REASON")
                elif c.get("loadable") is not True: rc.append("NOT_LOADABLE")
                if c.get("calibrationDigest")!=x.get("calibrationDigest"): rc.append("CALIBRATION_MISMATCH")
                if c.get("tokenizerDigest")!=x.get("tokenizerDigest"): rc.append("TOKENIZER_MISMATCH")
            outc.append({"name":c.get("name") if isinstance(c,dict) else "","status":"unsupported" if c.get("unsupportedReason") in allowed and not rc else ("invalid" if rc else "frozen"),
                         "inventory":inv if not any(z=="INVALID_INPUT" for z in rc) else [],"totalBytes":total if not any(z=="INVALID_INPUT" for z in rc) else None,
                         "packageDigest":pkg if not any(z=="INVALID_INPUT" for z in rc) else None,"reasonCodes":uniq_codes(rc)})
        outc.sort(key=lambda z:utf8key(z["name"]))
        out={"freezeId":fid,"candidates":outc}
        with LOCK:
            if fid in FREEZES:
                if compact(FREEZES[fid]["input"])==compact(x): return jsonify(FREEZES[fid]["output"])
                return jsonify({"error":"FREEZE_ID_CONFLICT"}),409
            FREEZES[fid]={"input":x,"output":out}
        return jsonify(out)
    fid=x.get("freezeId")
    if not isinstance(fid,str) or not fid or not isinstance(x.get("candidates"),list) or not isinstance(x.get("rows"),list) or not isinstance(x.get("policy"),dict): return bad_input()
    fr=FREEZES.get(fid)
    if not fr: 
        return jsonify({"freezeId":fid,"selected":None,"results":[],"packageManifest":None})
    frozen=fr["output"]["candidates"]
    if compact(x["candidates"])!=compact(frozen):
        # supplied lineage doesn't match; still return per-candidate results.
        pass
    p=x["policy"]; order=p.get("candidateOrder"); rows=x["rows"]; lat=x.get("latencies",{})
    if not isinstance(order,list) or len(order)!=len(set(order)) or not all(isinstance(n,str) for n in order): return jsonify({"freezeId":fid,"selected":None,"results":[],"packageManifest":None})
    res=[]; admitted=[]
    for c in frozen:
        name=c["name"]; rc=[]; agg=None; slices={}
        if not isinstance(c,dict) or c["status"] not in ("frozen","unsupported"): rc.append("INVALID_MANIFEST")
        if c["status"]=="unsupported": rc.append("NOT_FROZEN")
        # recompute manifest from stored inventory
        if c.get("inventory") is not None:
            recompute=[{"name":i.get("name"),"bytes":i.get("bytes"),"sha256":i.get("sha256")} for i in c["inventory"]]
            valid_inv=all(isinstance(i.get("name"),str) and isinstance(i.get("bytes"),int) and isinstance(i.get("sha256"),str) and
                          i["bytes"]>=0 and HEX64.fullmatch(i["sha256"] or "") for i in c["inventory"])
            if not valid_inv or sha_json(recompute)!=c.get("packageDigest") or sum(i["bytes"] for i in c["inventory"])!=c.get("totalBytes"): rc.append("INVALID_MANIFEST")
        if not safe_int(p.get("maxBytes")) or not finite_num(p.get("aggregateFloor")) or not 0<=p.get("aggregateFloor",0)<=1 or not finite_num(p.get("maxLatencyMs")) or p.get("maxLatencyMs")<0 or not isinstance(p.get("requiredSlices"),dict): rc.append("INVALID_POLICY")
        validpred=True
        if not rows: validpred=False; rc.append("INVALID_PREDICTIONS")
        else:
            for r in rows:
                pr=r.get("predictions",{}).get(name) if isinstance(r,dict) and isinstance(r.get("predictions"),dict) else None
                if r.get("label") not in (0,1) or pr not in (0,1) or not isinstance(r.get("slice"),str): validpred=False; break
            if not validpred: rc.append("INVALID_PREDICTIONS")
            else:
                agg=round(sum(r["label"]==r["predictions"][name] for r in rows)/len(rows),12)
                for sn,f in p["requiredSlices"].items():
                    sr=[r for r in rows if r["slice"]==sn]
                    if not sr: rc.append("MISSING_SLICE:"+sn)
                    else:
                        slices[sn]=round(sum(r["label"]==r["predictions"][name] for r in sr)/len(sr),12)
                        if slices[sn]<f: rc.append("SLICE_FLOOR:"+sn)
                if agg<p["aggregateFloor"]: rc.append("AGGREGATE_FLOOR")
        tb=c.get("totalBytes"); lv=lat.get(name) if isinstance(lat,dict) else None
        if not isinstance(tb,int) or tb<0: tb=None; rc.append("INVALID_MANIFEST")
        if not finite_num(lv) or lv<0: lv=None; rc.append("INVALID_LINEAGE")
        if tb is not None and tb>p.get("maxBytes",0): rc.append("SIZE_LIMIT")
        if lv is not None and lv>p.get("maxLatencyMs",0): rc.append("LATENCY_LIMIT")
        rc=uniq_codes(rc); ok=not rc
        rr={"name":name,"aggregate":agg,"slices":slices,"totalBytes":tb,"latencyMs":lv,"admitted":ok,"reasonCodes":rc}
        res.append(rr)
        if ok: admitted.append((tb,lv,order.index(name) if name in order else 10**9,name,c))
    res.sort(key=lambda r:(order.index(r["name"]) if r["name"] in order else 10**9,utf8key(r["name"])))
    winner=min(admitted,key=lambda z:(z[0],z[1],z[2])) if admitted else None
    return jsonify({"freezeId":fid,"selected":winner[3] if winner else None,"results":res,"packageManifest":winner[4] if winner else None})

# ---------------- 6. pipeline ----------------
DAG=["verify_data","prepare","train","evaluate","register","publish"]
INPUTS=["generation","checksum","canonicalData","prepareCode","prepareConfig","trainCode","trainConfig","runtime","evaluateCode","evaluateConfig","schemaDigest","publishConfig"]
def node_key(node, inp, artifacts):
    if node=="verify_data": arr=[inp["generation"],inp["checksum"]]
    elif node=="prepare": arr=[inp["canonicalData"],inp["prepareCode"],inp["prepareConfig"]]
    elif node=="train": arr=[artifacts.get("prepare"),inp["trainCode"],inp["trainConfig"],inp["runtime"]]
    elif node=="evaluate": arr=[artifacts.get("train"),inp["canonicalData"],inp["evaluateCode"],inp["evaluateConfig"]]
    elif node=="register": arr=[artifacts.get("evaluate"),inp["schemaDigest"]]
    else: arr=[artifacts.get("register"),inp["publishConfig"]]
    if any(v is None or not isinstance(v,str) or not v for v in arr): return None
    return sha_json(arr)
def dep_digests(node,inp,cache):
    if node=="verify_data": return {"generation":inp["generation"],"checksum":inp["checksum"]}
    if node=="prepare": return {"canonicalData":inp["canonicalData"],"prepareCode":inp["prepareCode"],"prepareConfig":inp["prepareConfig"]}
    if node=="train": return {"prepareArtifact":cache.get("prepare",""),"trainCode":inp["trainCode"],"trainConfig":inp["trainConfig"],"runtime":inp["runtime"]}
    if node=="evaluate": return {"trainArtifact":cache.get("train",""),"canonicalData":inp["canonicalData"],"evaluateCode":inp["evaluateCode"],"evaluateConfig":inp["evaluateConfig"]}
    if node=="register": return {"evaluateArtifact":cache.get("evaluate",""),"schemaDigest":inp["schemaDigest"]}
    return {"registerArtifact":cache.get("register",""),"publishConfig":inp["publishConfig"]}

@app.post("/pipeline")
def pipeline():
    x=json_obj(request)
    if not isinstance(x,dict) or not isinstance(x.get("session"),str) or not x["session"] or not safe_int(x.get("revision"),True) or not isinstance(x.get("inputs"),dict) or not isinstance(x.get("events"),list):
        return jsonify({"error":"INVALID_REQUEST"}),409
    s=x["session"]; rev=x["revision"]; inp=x["inputs"]
    if not all(isinstance(inp.get(k),str) and inp[k] for k in INPUTS): return jsonify({"error":"INVALID_REQUEST"}),409
    with LOCK:
        st=PIPELINES.setdefault(s,{"revision":rev,"input":inp,"events":{},"states":{},"cache":{},"history":[]})
        if rev<st["revision"]:
            pass
        elif rev>st["revision"]:
            st={"revision":rev,"input":inp,"events":{},"states":{},"cache":dict(st["cache"]),"history":[]}; PIPELINES[s]=st
        elif compact(st["input"])!=compact(inp):
            return jsonify({"error":"REVISION_CONFLICT"}),409
        accepted=[]; ignored=[]
        for e in x["events"]:
            if not isinstance(e,dict) or set(e.keys())!={"eventId","revision","node","attempt","status","key","artifactDigest","receiptId"}:
                return jsonify({"error":"INVALID_EVENT"}),409
            eid=e["eventId"]
            if eid in st["events"]:
                if compact(st["events"][eid])!=compact(e): return jsonify({"error":"EVENT_ID_CONFLICT"}),409
                ignored.append(eid); continue
            if e["revision"]!=rev or e["node"] not in DAG or not safe_int(e["attempt"],True) or e["status"] not in ("started","succeeded","retryable_failed","terminal_failed"):
                ignored.append(eid); continue
            # validate success/failure artifact and receipt
            if e["status"]=="succeeded" and (not isinstance(e["artifactDigest"],str) or not e["artifactDigest"]): ignored.append(eid); continue
            if e["status"]!="succeeded" and e["artifactDigest"] is not None: ignored.append(eid); continue
            if e["node"] in ("register","publish") and e["status"]=="succeeded" and e["receiptId"]!=f"receipt:{e['node']}:{e['key']}": ignored.append(eid); continue
            if e["node"] not in ("register","publish") and e["receiptId"] is not None: ignored.append(eid); continue
            artifacts=st["cache"]; key=node_key(e["node"],inp,artifacts)
            if e["key"]!=key: ignored.append(eid); continue
            idx=DAG.index(e["node"])
            if idx>0 and DAG[idx-1] not in artifacts and not (DAG[idx-1]==e["node"]): ignored.append(eid); continue
            state=st["states"].get(e["node"])
            if state is None:
                if e["status"]!="started" or e["attempt"]!=1: ignored.append(eid); continue
            elif state["status"]=="started":
                if e["attempt"]!=state["attempt"] or e["status"] not in ("succeeded","retryable_failed","terminal_failed"):
                    return jsonify({"error":"STATUS_CONFLICT"}),409
            elif state["status"]=="retryable_failed":
                if e["status"]!="started" or e["attempt"]!=state["attempt"]+1: return jsonify({"error":"STATUS_CONFLICT"}),409
            elif state["status"]=="succeeded":
                if e["status"]=="succeeded" and e["artifactDigest"]!=state["artifact"]: return jsonify({"error":"EVIDENCE_CONFLICT"}),409
                return jsonify({"error":"STATUS_CONFLICT"}),409
            elif state["status"]=="terminal_failed": return jsonify({"error":"STATUS_CONFLICT"}),409
            st["events"][eid]=e; accepted.append(eid); st["history"].append(eid)
            st["states"][e["node"]]={"status":e["status"],"attempt":e["attempt"],"eventId":eid,"artifact":e["artifactDigest"]}
            if e["status"]=="succeeded": st["cache"][e["node"]]=e["artifactDigest"]
        nodes=[]; blocked=False
        for i,n in enumerate(DAG):
            key=node_key(n,inp,st["cache"])
            deps=dep_digests(n,inp,st["cache"])
            deps["cacheKey"]=key
            state=st["states"].get(n)
            trig=[state["eventId"]] if state else []
            if n in st["cache"]: action,reason="reuse","CACHE_HIT"
            elif state and state["status"]=="started": action,reason="block","RUNNING"
            elif state and state["status"]=="terminal_failed": action,reason="block","TERMINAL_FAILURE"
            elif i>0 and DAG[i-1] not in st["cache"]:
                action,reason="block","UPSTREAM_TERMINAL" if st["states"].get(DAG[i-1],{}).get("status")=="terminal_failed" else "UPSTREAM_PENDING"
            elif state and state["status"]=="retryable_failed": action,reason="rerun","RETRYABLE_FAILURE"
            else: action,reason="rerun","CACHE_MISS"
            nodes.append({"node":n,"action":action,"reasonCodes":[reason],"dependencyDigests":deps,"triggeringEventIds":trig})
        return jsonify({"revision":rev,"acceptedEventIds":accepted,"ignoredEventIds":ignored,"nodes":nodes})

# ---------------- 7. verify bundle ----------------
def extract_card(readme):
    prefix='<!-- tds-model-card '
    suffix=' -->'
    matches=[]; pos=0
    while True:
        i=readme.find(prefix,pos)
        if i<0: break
        j=readme.find(suffix,i+len(prefix))
        if j<0:
            matches.append(None); break
        matches.append(readme[i+len(prefix):j]); pos=j+len(suffix)
    return matches

@app.post("/verify-bundle")
def verify_bundle():
    x=json_obj(request)
    if not isinstance(x,dict) or not isinstance(x.get("policy"),dict) or not isinstance(x.get("files"),dict): return bad_input()
    p=x["policy"]; f=x["files"]; v=[]
    if not isinstance(p.get("requiredSlices"),list) or not p["requiredSlices"] or len(set(p["requiredSlices"]))!=len(p["requiredSlices"]) or not all(isinstance(a,str) and a for a in p["requiredSlices"]) or not all(isinstance(p.get(k),str) and p[k] for k in ("license","intendedUse","limitations")):
        return bad_input()
    required=["README.md","training_manifest.json","evaluation.json","inventory.json","adapter_model.safetensors","adapter_config.json"]
    for n in required:
        if n not in f: v.append("MISSING_FILE:"+n)
        elif not isinstance(f[n],str): v.append("INVALID_FILE:"+n)
    for n in f:
        if not isinstance(n,str) or not isinstance(f[n],str): v.append("INVALID_FILE:"+str(n))
        if isinstance(n,str) and n.lower().endswith((".bin",".pt",".pth",".pkl",".pickle")): v.append("UNSAFE_WEIGHTS")
    inv=None
    if isinstance(f.get("inventory.json"),str):
        try: inv=json.loads(f["inventory.json"])
        except Exception: v.append("INVALID_JSON:inventory.json")
    if isinstance(inv,list):
        expected=[]
        for n in f:
            if n=="inventory.json": continue
            expected.append({"name":n,"bytes":len(f[n].encode("utf-8")),"sha256":sha_bytes(f[n].encode("utf-8"))})
        expected.sort(key=lambda z:utf8key(z["name"]))
        if inv!=expected: v.append("INVENTORY_MISMATCH")
        inventory_digest=sha_json(expected)
    else: inventory_digest=sha_json([])
    cfg=None
    try: cfg=json.loads(f.get("adapter_config.json",""))
    except Exception: v.append("INVALID_JSON:adapter_config.json")
    if not isinstance(cfg,dict) or not safe_int(cfg.get("r"),True) or not isinstance(cfg.get("target_modules"),list) or not cfg["target_modules"] or len(set(cfg["target_modules"]))!=len(cfg["target_modules"]) or not all(isinstance(a,str) and a for a in cfg["target_modules"]): v.append("INVALID_ADAPTER_CONFIG")
    man=None
    try: man=json.loads(f.get("training_manifest.json",""))
    except Exception: v.append("INVALID_JSON:training_manifest.json")
    fields=["task","datasetDigest","codeDigest","trainingConfigDigest","modelArtifactDigest","evaluationArtifactDigest"]
    if not isinstance(man,dict): v.append("INVALID_TRAINING_MANIFEST")
    else:
        if not HEX40.fullmatch(man.get("baseRevision","")): v.append("MUTABLE_BASE_REVISION")
        for k in fields:
            if not isinstance(man.get(k),str) or not man[k]: v.append("MISSING_MANIFEST_FIELD:"+k)
    if isinstance(man,dict):
        md=sha_bytes(f.get("adapter_model.safetensors","").encode("utf-8"))
        ed=sha_bytes(f.get("evaluation.json","").encode("utf-8"))
        if man.get("modelArtifactDigest")!=md: v.append("MODEL_ARTIFACT_MISMATCH")
        if man.get("evaluationArtifactDigest")!=ed: v.append("EVALUATION_DIGEST_MISMATCH")
    ev=None
    try: ev=json.loads(f.get("evaluation.json",""))
    except Exception: v.append("INVALID_JSON:evaluation.json")
    if not isinstance(ev,dict): v.append("INVALID_EVALUATION")
    else:
        if isinstance(man,dict) and ev.get("modelArtifactDigest")!=man.get("modelArtifactDigest"): v.append("EVALUATION_ARTIFACT_MISMATCH")
        if not finite_num(ev.get("aggregate")) or not 0<=ev.get("aggregate",0)<=1: v.append("INVALID_AGGREGATE")
        sl=ev.get("slices",{})
        if not isinstance(sl,dict): sl={}
        for n in p["requiredSlices"]:
            if n not in sl: v.append("MISSING_SLICE:"+n)
            elif not finite_num(sl[n]) or not 0<=sl[n]<=1: v.append("SLICE_RANGE:"+n)
    if isinstance(man,dict):
        cards=extract_card(f.get("README.md",""))
        if len(cards)==0: v.extend(["MODEL_CARD_COUNT","MISSING_MODEL_CARD"])
        elif len(cards)>1: v.append("MODEL_CARD_COUNT")
        else:
            try:
                card=json.loads(cards[0])
                if not isinstance(card,dict): raise ValueError()
                checks={"task":man.get("task"),"baseRevision":man.get("baseRevision"),"datasetDigest":man.get("datasetDigest"),
                        "modelArtifactDigest":man.get("modelArtifactDigest"),"license":p["license"],"intendedUse":p["intendedUse"],"limitations":p["limitations"]}
                if any(card.get(k)!=val for k,val in checks.items()): v.append("MODEL_CARD_MISMATCH")
            except Exception: v.append("INVALID_MODEL_CARD")
    v=uniq_codes(v)
    return jsonify({"decision":"admit" if not v else "reject","violations":v,"inventoryDigest":inventory_digest})

@app.get("/")
def index():
    return jsonify({"service":"ML governance routes","routes":["/build-corpus","/bqml","/promote","/adapt","/quantize","/pipeline","/verify-bundle"]})

if __name__=="__main__":
    import os
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT","10000")))

import { useState, useEffect, useRef, useCallback } from "react";

// ── Simulation constants ──────────────────────────────────────
const HIGHWAY_PATH = [
  [17.3850, 78.4867], [17.3870, 78.4920], [17.3895, 78.4975],
  [17.3920, 78.5030], [17.3945, 78.5085], [17.3970, 78.5140],
  [17.3990, 78.5195], [17.4010, 78.5250], [17.4035, 78.5305],
  [17.4060, 78.5360], [17.4085, 78.5415], [17.4110, 78.5470],
  [17.4135, 78.5525], [17.4155, 78.5580], [17.4175, 78.5635],
  [17.4200, 78.5690], [17.4225, 78.5745], [17.4250, 78.5800],
];

const CONGESTION_ZONES = [
  { lat: 17.3970, lng: 78.5140, radius: 400 },
  { lat: 17.4110, lng: 78.5470, radius: 350 },
];

function lerp(a, b, t) { return a + (b - a) * t; }
function lerpCoord(p1, p2, t) {
  return [lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t)];
}
function getPathPos(path, progress) {
  const p = Math.max(0, Math.min(0.9999, progress));
  const idx = Math.floor(p * (path.length - 1));
  const next = Math.min(idx + 1, path.length - 1);
  const local = p * (path.length - 1) - idx;
  return lerpCoord(path[idx], path[next], local);
}
function haversine([lat1, lon1], [lat2, lon2]) {
  const R = 6371000, dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
function inCongestion(pos) {
  return CONGESTION_ZONES.some(z => haversine(pos, [z.lat, z.lng]) < z.radius);
}

// ── Logistic Regression weights (from Python model) ──────────
const LR_WEIGHTS = {
  gap_m: -9.33, lead_vel_std: 0.57, lead_vel_mean: 0.40,
  newell_error: 0.27, ego_vel_mean: 0.24, ego_accel: 0.05,
  lead_accel: 0.01, relative_vel: -0.006, ego_vel_std: -0.0001,
};
const LR_BIAS = -0.5;
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function predictRisk(features) {
  let z = LR_BIAS;
  for (const [k, w] of Object.entries(LR_WEIGHTS)) z += w * (features[k] || 0);
  return sigmoid(z);
}

// ── Leaflet map component via vanilla JS ─────────────────────
function LeafletMap({ leadPos, egoPos, gap, riskProb, congestionZones, path }) {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const layersRef = useRef({});

  useEffect(() => {
    if (mapInstance.current) return;
    const L = window.L;
    if (!L) return;

    const map = L.map(mapRef.current, {
      center: [17.4030, 78.5200], zoom: 13,
      zoomControl: true, attributionControl: false,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: "CartoDB", maxZoom: 19,
    }).addTo(map);

    // Highway path
    const polyline = L.polyline(path, {
      color: "#00d4ff", weight: 4, opacity: 0.6, dashArray: "8 4"
    }).addTo(map);

    // Congestion zones
    congestionZones.forEach((z, i) => {
      const circle = L.circle([z.lat, z.lng], {
        radius: z.radius, color: "#ff6b35",
        fillColor: "#ff6b35", fillOpacity: 0.15, weight: 2, dashArray: "6 3"
      }).addTo(map);
      L.marker([z.lat, z.lng], {
        icon: L.divIcon({
          html: `<div style="background:#ff6b35;color:#000;font-size:9px;font-weight:700;
                  padding:2px 5px;border-radius:3px;white-space:nowrap;font-family:monospace">
                  CONGESTION ZONE ${i + 1}</div>`,
          className: "", iconAnchor: [40, -4]
        })
      }).addTo(map);
    });

    // Lead vehicle marker
    const leadIcon = L.divIcon({
      html: `<div id="lead-marker" style="
        width:36px;height:36px;background:#00d4ff;border-radius:50% 50% 50% 0;
        transform:rotate(-45deg);border:3px solid #fff;box-shadow:0 0 16px #00d4ff88;
        display:flex;align-items:center;justify-content:center;">
        <span style="transform:rotate(45deg);font-size:14px">🚗</span></div>`,
      className: "", iconAnchor: [18, 18]
    });
    const egoIcon = L.divIcon({
      html: `<div id="ego-marker" style="
        width:36px;height:36px;background:#ff4757;border-radius:50% 50% 50% 0;
        transform:rotate(-45deg);border:3px solid #fff;box-shadow:0 0 16px #ff475788;
        display:flex;align-items:center;justify-content:center;">
        <span style="transform:rotate(45deg);font-size:14px">🚙</span></div>`,
      className: "", iconAnchor: [18, 18]
    });

    layersRef.current.lead = L.marker(leadPos, { icon: leadIcon }).addTo(map);
    layersRef.current.ego  = L.marker(egoPos,  { icon: egoIcon }).addTo(map);

    // Gap line
    layersRef.current.gapLine = L.polyline([leadPos, egoPos], {
      color: "#ffd700", weight: 2, dashArray: "5 5", opacity: 0.8
    }).addTo(map);

    mapInstance.current = map;
  }, []);

  // Update markers on position change
  useEffect(() => {
    const layers = layersRef.current;
    if (!layers.lead || !layers.ego) return;
    layers.lead.setLatLng(leadPos);
    layers.ego.setLatLng(egoPos);
    layers.gapLine.setLatLngs([leadPos, egoPos]);

    // Update gap line color by risk
    const color = riskProb > 0.5 ? "#ff4757" : riskProb > 0.25 ? "#ffd700" : "#2ed573";
    layers.gapLine.setStyle({ color });

    // Update ego marker color
    const egoColor = riskProb > 0.5 ? "#ff4757" : riskProb > 0.25 ? "#ffa502" : "#2ed573";
    const egoEl = layers.ego.getElement();
    if (egoEl) {
      const div = egoEl.querySelector("div");
      if (div) {
        div.style.background = egoColor;
        div.style.boxShadow = `0 0 ${riskProb > 0.5 ? "24" : "12"}px ${egoColor}88`;
      }
    }
  }, [leadPos, egoPos, gap, riskProb]);

  return (
    <div ref={mapRef} style={{
      width: "100%", height: "100%", borderRadius: "0",
      filter: "saturate(1.1)"
    }} />
  );
}

// ── Main App ─────────────────────────────────────────────────
export default function V2VMapDashboard() {
  const [tick, setTick] = useState(0);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [leafletLoaded, setLeafletLoaded] = useState(false);

  // Vehicle states
  const leadProgress = useRef(0.02);
  const egoProgress  = useRef(0.00);
  const leadVelHist  = useRef(Array(30).fill(80));
  const egoVelHist   = useRef(Array(30).fill(75));
  const alertLog     = useRef([]);

  const [state, setState] = useState({
    leadPos: HIGHWAY_PATH[1], egoPos: HIGHWAY_PATH[0],
    leadVel: 80, egoVel: 75, gap: 40,
    riskProb: 0, riskLabel: "LOW RISK",
    collision_prob_pct: 0, alerts: [],
    newell_error: 0, relative_vel: 5,
    t: 0,
  });

  // Load Leaflet
  useEffect(() => {
    if (window.L) { setLeafletLoaded(true); return; }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
    document.head.appendChild(link);
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
    script.onload = () => setLeafletLoaded(true);
    document.head.appendChild(script);
  }, []);

  // Simulation loop
  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      const DT = 0.002 * speed;

      // Lead velocity
      const leadInCong = inCongestion(getPathPos(HIGHWAY_PATH, leadProgress.current));
      const targetLeadVel = leadInCong ? 20 + Math.random() * 10 : 85 + Math.random() * 10;
      const prevLead = leadVelHist.current[leadVelHist.current.length - 1];
      const newLeadVel = prevLead + (targetLeadVel - prevLead) * 0.05 + (Math.random() - 0.5) * 1.5;

      // Ego velocity — delayed reaction (Newell shift)
      const egoInCong = inCongestion(getPathPos(HIGHWAY_PATH, egoProgress.current));
      const newellPred = leadVelHist.current[leadVelHist.current.length - 10] || newLeadVel;
      const targetEgoVel = egoInCong
        ? newellPred * 0.7 + 10
        : newellPred + (Math.random() - 0.5) * 4;
      const prevEgo = egoVelHist.current[egoVelHist.current.length - 1];
      const newEgoVel = prevEgo + (targetEgoVel - prevEgo) * 0.04 + (Math.random() - 0.5) * 2;

      leadVelHist.current = [...leadVelHist.current.slice(-29), Math.max(10, Math.min(120, newLeadVel))];
      egoVelHist.current  = [...egoVelHist.current.slice(-29),  Math.max(10, Math.min(120, newEgoVel))];

      // Advance positions
      leadProgress.current = Math.min(0.98, leadProgress.current + DT * (newLeadVel / 3600) * 0.8);
      egoProgress.current  = Math.min(leadProgress.current - 0.005,
        egoProgress.current + DT * (newEgoVel / 3600) * 0.8);
      egoProgress.current = Math.max(0, egoProgress.current);

      const lPos = getPathPos(HIGHWAY_PATH, leadProgress.current);
      const ePos = getPathPos(HIGHWAY_PATH, egoProgress.current);
      const gapM = Math.max(2, haversine(lPos, ePos));

      const relVel = newLeadVel - newEgoVel;
      const nErr   = newEgoVel - newellPred;
      const lArr   = leadVelHist.current;
      const eArr   = egoVelHist.current;
      const lMean  = lArr.reduce((a,b)=>a+b,0)/lArr.length;
      const lStd   = Math.sqrt(lArr.reduce((a,b)=>a+(b-lMean)**2,0)/lArr.length);
      const eMean  = eArr.reduce((a,b)=>a+b,0)/eArr.length;
      const eStd   = Math.sqrt(eArr.reduce((a,b)=>a+(b-eMean)**2,0)/eArr.length);
      const lAcc   = lArr[lArr.length-1] - lArr[lArr.length-5];
      const eAcc   = eArr[eArr.length-1] - eArr[eArr.length-5];

      const norm = (v, m, s) => s > 0 ? (v - m) / s : 0;
      const features = {
        gap_m:          norm(gapM, 25, 15),
        lead_vel_mean:  norm(lMean, 70, 25),
        lead_vel_std:   norm(lStd, 10, 8),
        ego_vel_mean:   norm(eMean, 70, 25),
        ego_vel_std:    norm(eStd, 10, 8),
        relative_vel:   norm(relVel, 0, 15),
        newell_error:   norm(nErr, 0, 8),
        lead_accel:     norm(lAcc, 0, 5),
        ego_accel:      norm(eAcc, 0, 5),
      };
      const riskProb = predictRisk(features);
      const riskLabel = riskProb > 0.6 ? "HIGH RISK" : riskProb > 0.35 ? "MODERATE" : "LOW RISK";

      // Log alert
      if (riskProb > 0.6 && (alertLog.current.length === 0 ||
        alertLog.current[0].type !== "HIGH RISK")) {
        alertLog.current = [{
          type: "HIGH RISK",
          msg: `Gap ${gapM.toFixed(0)}m | Speed Δ ${Math.abs(relVel).toFixed(0)} km/h`,
          t: (tick * DT * 100).toFixed(0)
        }, ...alertLog.current.slice(0, 4)];
      } else if (riskProb > 0.35 && alertLog.current[0]?.type !== "HIGH RISK") {
        alertLog.current = [{
          type: "MODERATE",
          msg: `Gap ${gapM.toFixed(0)}m | Approaching congestion`,
          t: (tick * DT * 100).toFixed(0)
        }, ...alertLog.current.slice(0, 4)];
      }

      setTick(t => t + 1);
      setState({
        leadPos: lPos, egoPos: ePos,
        leadVel: Math.round(newLeadVel), egoVel: Math.round(newEgoVel),
        gap: Math.round(gapM), riskProb, riskLabel,
        collision_prob_pct: Math.round(riskProb * 100),
        alerts: [...alertLog.current],
        newell_error: nErr.toFixed(1),
        relative_vel: relVel.toFixed(1),
        t: tick,
      });
    }, 100);
    return () => clearInterval(interval);
  }, [running, speed, tick]);

  const riskColor = state.riskProb > 0.6 ? "#ff4757"
    : state.riskProb > 0.35 ? "#ffa502" : "#2ed573";

  return (
    <div style={{
      fontFamily: "'Courier New', monospace",
      background: "#0a0e1a",
      color: "#e0e8ff",
      height: "100vh",
      display: "grid",
      gridTemplateRows: "52px 1fr",
      gridTemplateColumns: "1fr 320px",
      overflow: "hidden",
    }}>
      {/* ── Header ── */}
      <div style={{
        gridColumn: "1 / -1",
        background: "linear-gradient(90deg, #0d1b2e, #0a1628)",
        borderBottom: "1px solid #1e3a5f",
        display: "flex", alignItems: "center",
        padding: "0 20px", gap: 16,
      }}>
        <div style={{
          background: "#00d4ff22", border: "1px solid #00d4ff44",
          borderRadius: 6, padding: "4px 12px",
          fontSize: 11, letterSpacing: 3, color: "#00d4ff", fontWeight: 700,
        }}>V2V</div>
        <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: 1 }}>
          Vehicle-to-Vehicle Communication Framework
        </span>
        <span style={{ fontSize: 10, color: "#4a7fa5", marginLeft: 4 }}>
          Rose-STL-Lab Architecture · Logistic Regression · Hyderabad NH-163
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{
            background: `${riskColor}22`, border: `1px solid ${riskColor}66`,
            color: riskColor, padding: "3px 10px", borderRadius: 4,
            fontSize: 11, fontWeight: 700, letterSpacing: 1,
            animation: state.riskProb > 0.6 ? "pulse 0.8s infinite" : "none",
          }}>
            {state.riskLabel}
          </span>
          <button onClick={() => setRunning(r => !r)} style={{
            background: running ? "#ff475722" : "#2ed57322",
            border: `1px solid ${running ? "#ff4757" : "#2ed573"}`,
            color: running ? "#ff4757" : "#2ed573",
            padding: "4px 14px", borderRadius: 4, cursor: "pointer",
            fontSize: 11, fontWeight: 700, letterSpacing: 1,
          }}>{running ? "⏸ PAUSE" : "▶ RUN"}</button>
          <select value={speed} onChange={e => setSpeed(+e.target.value)} style={{
            background: "#0d1b2e", border: "1px solid #1e3a5f",
            color: "#e0e8ff", padding: "4px 8px", borderRadius: 4,
            fontSize: 11, cursor: "pointer",
          }}>
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={3}>3×</option>
          </select>
        </div>
      </div>

      {/* ── Map ── */}
      <div style={{ position: "relative", overflow: "hidden" }}>
        {leafletLoaded ? (
          <LeafletMap
            leadPos={state.leadPos} egoPos={state.egoPos}
            gap={state.gap} riskProb={state.riskProb}
            congestionZones={CONGESTION_ZONES}
            path={HIGHWAY_PATH}
          />
        ) : (
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            height: "100%", color: "#4a7fa5", fontSize: 13
          }}>Loading map...</div>
        )}

        {/* Risk probability overlay */}
        <div style={{
          position: "absolute", bottom: 16, left: 16,
          background: "#0a0e1acc", backdropFilter: "blur(8px)",
          border: `1px solid ${riskColor}44`,
          borderRadius: 8, padding: "10px 16px", minWidth: 200,
        }}>
          <div style={{ fontSize: 9, color: "#4a7fa5", letterSpacing: 2, marginBottom: 6 }}>
            COLLISION PROBABILITY
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              flex: 1, height: 8, background: "#1a2744",
              borderRadius: 4, overflow: "hidden",
            }}>
              <div style={{
                width: `${state.collision_prob_pct}%`,
                height: "100%",
                background: `linear-gradient(90deg, #2ed573, #ffa502, #ff4757)`,
                borderRadius: 4,
                transition: "width 0.3s ease",
              }} />
            </div>
            <span style={{ color: riskColor, fontWeight: 700, fontSize: 16, minWidth: 44 }}>
              {state.collision_prob_pct}%
            </span>
          </div>
        </div>

        {/* Legend */}
        <div style={{
          position: "absolute", top: 12, right: 12,
          background: "#0a0e1acc", backdropFilter: "blur(8px)",
          border: "1px solid #1e3a5f",
          borderRadius: 8, padding: "10px 14px", fontSize: 11,
        }}>
          <div style={{ color: "#4a7fa5", fontSize: 9, letterSpacing: 2, marginBottom: 8 }}>LEGEND</div>
          {[
            ["🚗", "#00d4ff", "Lead Vehicle"],
            ["🚙", "#ff4757", "Ego Vehicle"],
            ["━━", "#ffd700", "V2V Link"],
            ["◉", "#ff6b35", "Congestion Zone"],
          ].map(([icon, color, label]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
              <span style={{ color, fontSize: 13 }}>{icon}</span>
              <span style={{ color: "#8aa8cc" }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Sidebar ── */}
      <div style={{
        background: "#0d1421",
        borderLeft: "1px solid #1e3a5f",
        overflowY: "auto",
        padding: 16,
        display: "flex", flexDirection: "column", gap: 14,
      }}>

        {/* Vehicle Stats */}
        {[
          {
            label: "LEAD VEHICLE", color: "#00d4ff", icon: "🚗",
            vel: state.leadVel, lat: state.leadPos[0]?.toFixed(4),
            lng: state.leadPos[1]?.toFixed(4),
          },
          {
            label: "EGO VEHICLE", color: "#ff4757", icon: "🚙",
            vel: state.egoVel, lat: state.egoPos[0]?.toFixed(4),
            lng: state.egoPos[1]?.toFixed(4),
          },
        ].map(v => (
          <div key={v.label} style={{
            background: `${v.color}08`,
            border: `1px solid ${v.color}33`,
            borderRadius: 8, padding: 12,
          }}>
            <div style={{
              fontSize: 9, color: v.color, letterSpacing: 2,
              marginBottom: 8, fontWeight: 700,
              display: "flex", alignItems: "center", gap: 6
            }}>
              {v.icon} {v.label}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[
                ["SPEED", `${v.vel} km/h`],
                ["LAT", v.lat],
                ["LNG", v.lng],
              ].map(([k, val]) => (
                <div key={k}>
                  <div style={{ fontSize: 8, color: "#4a7fa5", letterSpacing: 1 }}>{k}</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: v.color }}>{val}</div>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* V2V Packet Data */}
        <div style={{
          background: "#0a1628",
          border: "1px solid #1e3a5f",
          borderRadius: 8, padding: 12,
        }}>
          <div style={{ fontSize: 9, color: "#4a7fa5", letterSpacing: 2, marginBottom: 10, fontWeight: 700 }}>
            📡 V2V PACKET
          </div>
          {[
            ["GAP DISTANCE", `${state.gap} m`, state.gap < 15 ? "#ff4757" : state.gap < 30 ? "#ffa502" : "#2ed573"],
            ["RELATIVE VEL", `${state.relative_vel} km/h`, +state.relative_vel < -15 ? "#ff4757" : "#e0e8ff"],
            ["NEWELL ERROR", `${state.newell_error} km/h`, Math.abs(+state.newell_error) > 10 ? "#ffa502" : "#e0e8ff"],
          ].map(([label, val, color]) => (
            <div key={label} style={{
              display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 8,
            }}>
              <span style={{ fontSize: 10, color: "#4a7fa5" }}>{label}</span>
              <span style={{ fontSize: 12, fontWeight: 700, color }}>{val}</span>
            </div>
          ))}
          <div style={{
            marginTop: 8, padding: "8px 0 0",
            borderTop: "1px solid #1e3a5f",
            display: "flex", justifyContent: "space-between", alignItems: "center"
          }}>
            <span style={{ fontSize: 10, color: "#4a7fa5" }}>RISK SCORE</span>
            <span style={{
              fontSize: 14, fontWeight: 700, color: riskColor,
              fontVariantNumeric: "tabular-nums"
            }}>
              {(state.riskProb * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Mini velocity chart */}
        <div style={{
          background: "#0a1628", border: "1px solid #1e3a5f",
          borderRadius: 8, padding: 12,
        }}>
          <div style={{ fontSize: 9, color: "#4a7fa5", letterSpacing: 2, marginBottom: 8, fontWeight: 700 }}>
            VELOCITY HISTORY
          </div>
          <svg width="100%" height="60" viewBox="0 0 288 60">
            {/* Lead sparkline */}
            {leadVelHist.current.length > 1 && (
              <polyline
                points={leadVelHist.current.map((v, i) =>
                  `${(i / 29) * 288},${60 - (v / 130) * 60}`).join(" ")}
                fill="none" stroke="#00d4ff" strokeWidth="1.5" opacity="0.8"
              />
            )}
            {/* Ego sparkline */}
            {egoVelHist.current.length > 1 && (
              <polyline
                points={egoVelHist.current.map((v, i) =>
                  `${(i / 29) * 288},${60 - (v / 130) * 60}`).join(" ")}
                fill="none" stroke="#ff4757" strokeWidth="1.5" opacity="0.8"
              />
            )}
          </svg>
          <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
            {[["Lead", "#00d4ff"], ["Ego", "#ff4757"]].map(([l, c]) => (
              <div key={l} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9 }}>
                <div style={{ width: 16, height: 2, background: c, borderRadius: 1 }} />
                <span style={{ color: "#4a7fa5" }}>{l}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Alert Log */}
        <div style={{
          background: "#0a1628", border: "1px solid #1e3a5f",
          borderRadius: 8, padding: 12, flex: 1,
        }}>
          <div style={{ fontSize: 9, color: "#4a7fa5", letterSpacing: 2, marginBottom: 8, fontWeight: 700 }}>
            ⚠ ALERT LOG
          </div>
          {state.alerts.length === 0 ? (
            <div style={{ fontSize: 10, color: "#2a3f5f", textAlign: "center", padding: "16px 0" }}>
              No alerts. System nominal.
            </div>
          ) : state.alerts.map((a, i) => (
            <div key={i} style={{
              marginBottom: 8, padding: "6px 8px",
              background: a.type === "HIGH RISK" ? "#ff475710" : "#ffa50210",
              border: `1px solid ${a.type === "HIGH RISK" ? "#ff475733" : "#ffa50233"}`,
              borderRadius: 5,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                <span style={{
                  fontSize: 9, fontWeight: 700, letterSpacing: 1,
                  color: a.type === "HIGH RISK" ? "#ff4757" : "#ffa502"
                }}>{a.type}</span>
                <span style={{ fontSize: 9, color: "#2a4f6f" }}>t={a.t}s</span>
              </div>
              <div style={{ fontSize: 9, color: "#8aa8cc" }}>{a.msg}</div>
            </div>
          ))}
        </div>

      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }
      `}</style>
    </div>
  );
}
/* Display-only scope from the actual completed report and scored clip metadata. */
"use strict";
function evaluationScopeText(report, clips) {
    const positive = clips.filter(c => c.kind === "positive");
    const originals = new Set();
    for (const clip of positive) {
        if (clip.calls && clip.calls.length) {
            for (const call of clip.calls) {
                originals.add(String(call.original_utterance_id || clip.original_id || clip.sample_id));
            }
        } else {
            originals.add(String(clip.original_id || clip.sample_id));
        }
    }
    const noiseVariants = positive.filter(c => c.category !== "clean").length;
    const summary = (report.models || []).find(m => m.negative && Number.isFinite(m.negative.hours));
    const hours = summary ? summary.negative.hours : null;
    const synthetic = positive.length > 0 && positive.every(c => c.provenance === "qwen_tts");
    const count = n => n.toLocaleString("ko-KR");
    let durationGate = "장시간 기준 미확인";
    if (report.stage === "final" && typeof report.final_negative_duration_passed === "boolean") {
        durationGate = report.final_negative_duration_passed ? "장시간 기준 통과" : "장시간 기준 미달";
    } else if ((report.adoption && report.adoption.reasons || []).some(reason =>
        typeof reason === "string" && reason.startsWith("Development FP duration below "))) {
        durationGate = "장시간 기준 미달";
    } else if (hours !== null && report.adoption && report.adoption.passed === true) {
        durationGate = "장시간 기준 통과";
    }
    const time = hours === null ? "시간 미측정" : hours < 1
        ? `${(hours * 60).toFixed(2)}분` : `${hours.toFixed(2)}시간`;
    const parts = [
        report.stage === "final" ? "동결 모델 Final 평가" : "개발 평가 (Final 아님)",
        `${synthetic ? "합성 " : ""}호출 원본 ${count(originals.size)}개`,
        `소음 변형 ${count(noiseVariants)}개`,
        `비호출 평가 ${time}`,
        durationGate,
    ];
    if (report.smoke_test_only) parts.push("기능 시험용 결과");
    return parts.join(" · ");
}
function evaluationNoiseCoverageText(clips) {
    const labels = {environment: "환경음", music: "음악·전자음", competing_speech: "겹치는 한국어 음성"};
    const parts = [];
    for (const [category, label] of Object.entries(labels)) {
        const rows = clips.filter(c => c.kind === "positive" && c.category === category);
        if (!rows.length) continue;
        const sources = new Set(rows.map(c => c.noise_source_group_id || c.noise_source_id).filter(Boolean));
        const types = new Set(rows.map(c => c.noise_eval_type).filter(Boolean));
        const sourceText = sources.size ? `원본 ${sources.size.toLocaleString("ko-KR")}개` : "원본 수 미기록";
        parts.push(`${label} ${sourceText} · ${types.size ? types.size.toLocaleString("ko-KR") + "종" : "세부 종류 미기록"}`);
    }
    return parts.length ? "시험에 사용한 배경 음원: " + parts.join(" / ") : "시험에 사용한 배경 음원: 미측정";
}
function evaluationBackgroundTypeText(clip) {
    const types = {rain: "비", wind: "바람", vacuum_cleaner: "청소기", washing_machine: "세탁기", engine: "엔진", train: "기차",
        keyboard_typing: "키보드", door_wood_knock: "문 두드림", glass_breaking: "유리 충격·파손", footsteps: "발걸음",
        clock_alarm: "알람", clock_tick: "시계음", car_horn: "경적", siren: "사이렌", church_bells: "종소리",
        music_instrumental_piano_drums: "피아노·드럼", music_instrumental_guitar_drums: "기타·드럼", music_singing: "보컬 멜로디", music_vocal_pad: "보컬 음악",
        korean_read_speech: "한국어 낭독", korean_speech: "한국어 음성", korean_market_conversation: "한국어 시장 대화",
        korean_restaurant_conversation: "한국어 식당 대화", korean_subway_conversation: "한국어 지하철 대화",
        korean_street_conversation: "한국어 거리 대화", korean_transit_announcement: "한국어 교통 안내방송"};
    return clip.noise_eval_type ? types[clip.noise_eval_type] || clip.noise_eval_type : "";
}
if (typeof module !== "undefined" && module.exports) module.exports = {evaluationScopeText, evaluationNoiseCoverageText, evaluationBackgroundTypeText};

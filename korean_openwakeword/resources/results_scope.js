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
    const summary = (report.models || []).find(m => m.negative && Number.isFinite(m.negative.hours));
    const hours = summary ? summary.negative.hours : null;
    const synthetic = positive.length > 0 && positive.every(c => c.provenance === "qwen_tts");
    const count = n => n.toLocaleString("ko-KR");
    const time = hours === null ? "시간 정보 없음" : hours < 1
        ? `${(hours * 60).toFixed(2)}분` : `${hours.toFixed(2)}시간`;
    const parts = [
        `${synthetic ? "합성 호출어" : "호출어 원본"} ${count(originals.size)}개`,
        `호출 시험 ${count(positive.length)}회`,
        `오작동 시험 ${time}`,
    ];
    if (report.smoke_test_only) parts.push("간단한 동작 시험");
    return parts.join(" · ");
}
function evaluationNoiseCoverageText(clips) {
    const labels = {environment: "환경소음", music: "음악·전자음", competing_speech: "다른 사람 말소리"};
    const parts = [];
    for (const [category, label] of Object.entries(labels)) {
        const rows = clips.filter(c => c.kind === "positive" && c.category === category);
        if (!rows.length) continue;
        const sources = new Set(rows.map(c => c.noise_source_group_id || c.noise_source_id).filter(Boolean));
        const types = new Set(rows.map(c => c.noise_eval_type).filter(Boolean));
        const sourceText = sources.size ? `원본 ${sources.size.toLocaleString("ko-KR")}개` : "음원 수 정보 없음";
        parts.push(`${label} ${sourceText} · ${types.size ? types.size.toLocaleString("ko-KR") + "종" : "종류 정보 없음"}`);
    }
    return parts.length ? "시험에 사용한 배경 음원: " + parts.join(" / ") : "시험에 사용한 배경 음원: 정보 없음";
}
function evaluationBackgroundTypeText(clip) {
    const types = {rain: "비", wind: "바람", vacuum_cleaner: "청소기", washing_machine: "세탁기", engine: "엔진", train: "기차",
        keyboard_typing: "키보드", door_wood_knock: "문 두드림", glass_breaking: "유리 충격·파손", footsteps: "발걸음",
        clock_alarm: "알람", clock_tick: "시계 전자·기계음", car_horn: "경적", siren: "사이렌", church_bells: "종소리",
        music_instrumental_piano_drums: "피아노·드럼 연주", music_instrumental_guitar_drums: "기타·드럼 연주", music_singing: "보컬 멜로디", music_vocal_pad: "보컬 패드 음악",
        korean_read_speech: "한국어 낭독", korean_speech: "한국어 음성", korean_market_conversation: "한국어 시장 대화",
        korean_restaurant_conversation: "한국어 식당 대화", korean_subway_conversation: "한국어 지하철 대화",
        korean_street_conversation: "한국어 거리 대화", korean_transit_announcement: "한국어 교통 안내방송",
        korean_natural_conversation: "한국어 자연 대화", korean_conversation: "한국어 대화",
        korean_broadcast: "한국어 방송", korean_overlapping_speech: "겹치는 한국어 음성",
        dog: "개 짖음", cat: "고양이", chirping_birds: "새소리", crying_baby: "아기 울음",
        sneezing: "재채기", clapping: "박수", breathing: "숨소리", coughing: "기침", laughing: "웃음", snoring: "코골이",
        pouring_water: "물 붓기", toilet_flush: "변기 물 내림", water_drops: "물방울", can_opening: "캔 따기",
        mouse_click: "마우스 클릭", door_wood_creaks: "문 삐걱거림", airplane: "비행기", helicopter: "헬리콥터",
        thunderstorm: "천둥", sea_waves: "파도", music_jazz: "재즈", music_techno: "테크노", music_hiphop: "힙합",
        music_rock_vocals: "보컬 록", music_rock: "록", music_cinematic_orchestral: "영화풍 관현악",
        music_house: "하우스 음악", music_orchestral_loop: "관현악 루프"};
    const label = typeof clip.noise_eval_label === "string" ? clip.noise_eval_label.trim() : "";
    if (/[가-힣]/.test(label) && !label.includes("�")) return label;
    if (!clip.noise_eval_type) return "";
    if (types[clip.noise_eval_type]) return types[clip.noise_eval_type];
    const simple = {alarm:"알람", bell:"종소리", bicycle_bell:"자전거 벨", chime:"알림음", doorbell:"초인종", printer:"프린터", siren:"사이렌", telephone:"전화 소리", tick:"시계 소리", "tick-tock":"시계 소리",
        bird:"새소리", car:"자동차", cutlery_and_silverware:"식기 소리", door:"문소리", fireworks:"불꽃놀이", glass:"유리 소리", insect:"벌레 소리", pour:"물 붓는 소리", power_tool:"전동 공구", rain:"비", rattle:"덜컹거림", thump_and_thud:"둔탁한 충격음", thunder:"천둥", thunderstorm:"천둥과 비", tools:"공구 소리", train:"기차", walk_and_footsteps:"발걸음", water:"물소리", waves_and_surf:"파도",
        blues:"블루스", electronica:"일렉트로니카", funk:"펑크 음악", hiphop:"힙합", indierock:"인디 록", jazz:"재즈", pop:"팝 음악", westernart:"클래식 음악"};
    const canonical = clip.noise_eval_type.replace(/^(environment|electronic|music)_/, "");
    if (simple[canonical]) return simple[canonical];
    const languages = {danish:"덴마크어",english:"영어",french:"프랑스어",german:"독일어",hebrew:"히브리어",italian:"이탈리아어",japanese:"일본어",latin:"라틴어",polish:"폴란드어",portuguese:"포르투갈어",russian:"러시아어",spanish:"스페인어",tagalog:"타갈로그어"};
    if (clip.noise_eval_type.startsWith("read_speech_") && languages[clip.noise_eval_type.slice(12)]) return languages[clip.noise_eval_type.slice(12)] + " 낭독";
    if (clip.noise_eval_type === "korean_spontaneous_dialogue") return "한국어 대화·인터뷰";
    return {environment:"환경소음",music:"음악·전자음",competing_speech:"다른 사람 말소리"}[clip.category] || "기타 배경";
}
if (typeof module !== "undefined" && module.exports) module.exports = {evaluationScopeText, evaluationNoiseCoverageText, evaluationBackgroundTypeText};

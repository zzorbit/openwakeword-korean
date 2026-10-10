# 저장 정책

데이터·feature·모델·평가 WAV·다운로드·임시 파일은 사용자가 지정한 작업 드라이브에 저장한다.
API 기본 모델/임시 캐시는 `project.json` 옆 `cache`이며 `KOWW_CACHE_ROOT`로 공유 경로를 명시할 수 있다.
PowerShell 설치 전에 `. .\scripts\Use-ProjectStorage.ps1`을 실행하면 해당 체크아웃의 `storage/runtime_cache`를 사용한다.
환경 변경은 현재 셸/자식 프로세스 또는 API 호출 범위에만 적용되고 원래 설정은 복원한다.
`koww storage --project 프로젝트/project.json`으로 확인한다. `.codex`는 대량 자료 저장소로 쓰지 않는다.
SSD 복사는 자동 동작이 아니며 사용자가 선택한 경우 용량·기간·원본 보존·해시·정리 계획을 정한다.
캐시를 옮길 때는 원본 보존→복사→전체 SHA256→참조 검증→전환 순서다. 실행 중이거나 미검증된 입력은 삭제하지 않는다.

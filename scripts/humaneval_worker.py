import json
import signal
import sys
import traceback


EXECUTION_TIMEOUT_SECONDS = 15


class EvaluationTimeoutError(Exception):
    """Raised when generated code exceeds the execution timeout."""


def handle_timeout(signum, frame):
    raise EvaluationTimeoutError(
        f"Generated code exceeded "
        f"{EXECUTION_TIMEOUT_SECONDS} seconds"
    )


def main() -> None:
    payload = json.load(sys.stdin)

    generated_code = payload.get("generated_code", "")
    test_code = payload.get("test", "")
    entry_point = payload.get("entry_point", "")

    result = {
        "passed": False,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }

    namespace = {}

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(EXECUTION_TIMEOUT_SECONDS)

    try:
        # 모델이 생성한 코드 실행
        exec(generated_code, namespace)

        if entry_point not in namespace:
            raise NameError(
                f"Entry point '{entry_point}' was not found"
            )

        candidate = namespace[entry_point]

        # 공식 HumanEval 테스트 코드 로드
        exec(test_code, namespace)

        if "check" not in namespace:
            raise NameError(
                "Official HumanEval check function was not found"
            )

        # 공식 HumanEval 테스트 실행
        namespace["check"](candidate)

        result["passed"] = True

    except EvaluationTimeoutError as error:
        result["error_type"] = "TimeoutExpired"
        result["error_message"] = str(error)

    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        result["traceback"] = traceback.format_exc(limit=5)

    finally:
        signal.alarm(0)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
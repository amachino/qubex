"""Report pytest JUnit results and reject ordinary skips in CI."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    """Print test counts, append an Actions Summary, and fail on skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Pytest JUnit XML report")
    report = parser.parse_args().report
    root = ET.parse(report).getroot()  # noqa: S314 - Generated locally by pytest.
    cases = list(root.iter("testcase"))
    skipped = []
    xfailed = 0
    for case in cases:
        for skip in case.findall("skipped"):
            if skip.get("type") == "pytest.xfail":
                xfailed += 1
            else:
                skipped.append(
                    (
                        f"{case.get('classname', '')}::{case.get('name', '')}",
                        skip.get("message", ""),
                    )
                )
    counts = {
        "Passed": sum(
            not any(
                case.find(tag) is not None for tag in ("failure", "error", "skipped")
            )
            for case in cases
        ),
        "Failed": len(root.findall(".//testcase/failure")),
        "Errors": len(root.findall(".//testcase/error")),
        "Skipped": len(skipped),
        "Xfailed": xfailed,
    }
    version = sys.version.split()[0]
    print(f"pytest results · Python {version}\n")
    for name, count in counts.items():
        print(f"  {count:>4} {name.lower()}")
    print()
    if skipped:
        suffix = "s" if len(skipped) != 1 else ""
        print(f"Skip check failed: {len(skipped)} skipped test{suffix}.")
    else:
        print("Skip check passed: no skipped tests.")

    verdict = "✗ Skipped tests detected" if skipped else "✓ No skipped tests"
    summary = "\n".join(
        [
            f"### Test results — Python {version}",
            "",
            "| Result | Count |",
            "| --- | ---: |",
            *(f"| {name} | {count} |" for name, count in counts.items()),
            "",
            f"**{verdict}**",
            "",
        ]
    )
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as output:
            output.write(summary)
    if skipped:
        print()
        for test, reason in skipped:
            print(f"  {test}\n    {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

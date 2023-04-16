import sys
import os
import subprocess
import json

def get_returns_of_command(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()

    return {
        "stdout": stdout.decode("utf-8"),
        "stderr": stderr.decode("utf-8"),
        "code": proc.returncode,
    }

def test_file(file_path):
    get_returns_of_command(["python", "onicode.py", "file_path"])

    clean_path = os.path.splitext(file_path)[0]
    returns = get_returns_of_command([clean_path])

    if not os.path.exists(clean_path + ".json"):
        print(f"[Note]: File {file_path} does not have record file. skiping...")
        return

    with open(clean_path + ".json") as record:
        recored_rets = json.load(record)

        print(f"[Note]: Testing {file_path}...")

        if returns["stdout"] != recored_rets["stdout"]:
            print(f"[Error]: Test for {file_path} has falied")

            print(f"[Expect]: STDOUT:")
            print(recored_rets["stdout"])
            print(f"[Received]: STDOUT:")
            print(returns["stdout"])

            exit(1)
        if returns["stderr"] != recored_rets["stderr"]:
            print(f"[Error]: Test for {file_path} has falied")

            print(f"[Expect]: STDERR:")
            print(recored_rets["stderr"])
            print(f"[Received]: STDERR:")
            print(returns["stderr"])

            exit(1)
        if returns["code"] != recored_rets["code"]:
            print(f"[Error]: Test for {file_path} has falied")

            print(f"[Expect]: CODE:")
            print(recored_rets["code"])
            print(f"[Received]: CODE:")
            print(returns["code"])

            exit(1)

        print(f"[Success]: Test for {file_path} has passed")

def record_file(file_path):
    get_returns_of_command(["python", "onicode.py", file_path])

    clean_path = os.path.splitext(file_path)[0]

    returns = get_returns_of_command([clean_path])

    print(f"[Note]: Getting output of {file_path}...")
    with open(clean_path + ".json", "w") as record:
        json.dump(returns, record)


def test_files(file_path):
    if file_path != None:
        test_file(file_path)
        return

    for root, dirs, files in os.walk("tests/"):
        for file in files:
            if file.endswith(".oni"):
                test_file(os.path.join(root, file))

def record_files(file_path):
    if file_path != None:
        record_file(file_path)
        return

    for root, dirs, files in os.walk("tests/"):
        for file in files:
            if file.endswith(".oni"):
                record_file(os.path.join(root, file))

def usage(fd, program):
    print(f"[Usage]: {program} <ARGS>", file=fd)
    print(f"  [ARGS]:", file=fd)
    print(f"    test                           - test all files", file=fd)
    print(f"    record                          - record all files", file=fd)

def main():
    program_name, *argv = sys.argv

    if len(argv) <= 0:
        usage(sys.stderr, program_name)
        print("[Error]: Arguements is not provided")

    test = False
    record = False

    file = None

    while len(argv) > 0:
        arg, *argv = argv

        match arg:
            case "test": 
                test = True
            case "record":
                record = True
            case _:
                if file == None:
                    file = arg

                    continue
                usage(sys.stderr, program_name)
                print(f"[Error]: Unknown argument: {arg}")
                exit(1)

    if record:
        record_files(file)

    if test:
        test_files(file)


if __name__ == "__main__":
    main()

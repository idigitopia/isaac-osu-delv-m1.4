import argparse
import os

# from runner import Runner
from go2_runner import Runner


def main(network, policy_path, no_log):
    runner = Runner(policy_path, network, no_log)
    runner.run_main_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Go2 SDK agent.")
    parser.add_argument("--network", type=str, default="eth0", help="Internal computer network name.")
    parser.add_argument("--path", type=str, required=True, help="The complete path to the policy to .pt.")
    parser.add_argument("--no_log", action="store_true", help="Whether to log data or not. By default is False.")
    args = parser.parse_args()

    args.path = os.path.abspath(args.path)

    main(args.network, args.path, args.no_log)

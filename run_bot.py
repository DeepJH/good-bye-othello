import argparse
import sys
import time
from othello_agent.controllers.adb import AdbController
from othello_agent.vision.calibration import BoardConfig
from othello_agent.vision.recognizer import BoardRecognizer
from othello_agent.model_adapter import AlphaZeroModelAdapter
from othello_agent.orchestrator import GameOrchestrator, StepStatus
from othello_agent.protocols import Player

def main():
    parser = argparse.ArgumentParser(description="good-bye-othello: ADB AI Bot Runner")
    parser.add_argument('--config', type=str, default='config/board_config.json', help='Path to board ROI config')
    parser.add_argument('--color', type=str, default='auto', choices=['auto', 'black', 'white'], help='AI player color')
    parser.add_argument('--sims', type=int, default=25, help='MCTS simulation iterations per move')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds')
    parser.add_argument('--serial', type=str, default=None, help='ADB device serial')
    args = parser.parse_args()

    controller = AdbController(serial=args.serial)
    if not controller.is_connected():
        print("[ERROR] No ADB device connected. Please connect your Android phone/emulator and enable USB debugging.")
        print("Tip: Run 'adb devices' in terminal to verify connection.")
        sys.exit(1)

    try:
        config = BoardConfig.load(args.config)
    except FileNotFoundError:
        print(f"[ERROR] Config file '{args.config}' not found. Please calibrate or provide a valid config file.")
        sys.exit(1)

    recognizer = BoardRecognizer(config)
    model = AlphaZeroModelAdapter(board_size=config.rows, num_sims=args.sims)

    ai_player = None
    if args.color == 'black':
        ai_player = Player.BLACK
    elif args.color == 'white':
        ai_player = Player.WHITE

    orchestrator = GameOrchestrator(controller, recognizer, model, ai_player=ai_player)
    print("=======================================================")
    print(" Othello ADB AI Bot Started")
    print(f" Board: {config.rows}x{config.cols} | AI Color: {args.color} | Sims: {args.sims}")
    print(" Press Ctrl+C to stop.")
    print("=======================================================")

    try:
        while True:
            report = orchestrator.step()
            now_str = time.strftime('%H:%M:%S')
            print(f"[{now_str}] Status: {report.status.value} - {report.message}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")

if __name__ == "__main__":
    main()

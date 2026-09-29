"""
Calibration helper for setting up board coordinates (ROI) for any Android Othello App.
"""
import argparse
import sys
from PIL import Image, ImageDraw
from othello_agent.controllers.adb import AdbController
from othello_agent.vision.calibration import BoardConfig

def main():
    parser = argparse.ArgumentParser(description="good-bye-othello: Calibrate Board ROI")
    parser.add_argument('--capture', type=str, default=None, help='Capture screenshot from ADB and save to this filename')
    parser.add_argument('--image', type=str, default=None, help='Input screenshot image file to calibrate')
    parser.add_argument('--roi', type=str, default=None, help='TopLeftX,TopLeftY,BottomRightX,BottomRightY (e.g. 40,600,1040,1600)')
    parser.add_argument('--rows', type=int, default=8, help='Board rows (default: 8)')
    parser.add_argument('--cols', type=int, default=8, help='Board cols (default: 8)')
    parser.add_argument('--save', type=str, default='config/board_config.json', help='Path to save JSON config')
    parser.add_argument('--preview', type=str, default='config/calibration_preview.png', help='Save visual preview with grid overlay')
    args = parser.parse_args()

    # Step 1: Capture screenshot if requested
    if args.capture:
        adb = AdbController()
        if not adb.is_connected():
            print("[ERROR] No ADB device connected. Please plug in your phone or start emulator.")
            sys.exit(1)
        img = adb.capture_screen()
        img.save(args.capture)
        print(f"[OK] Screenshot captured and saved to '{args.capture}' (Size: {img.size[0]}x{img.size[1]})")
        if not args.image:
            args.image = args.capture

    if not args.image:
        print("Please provide --image <screenshot.png> or --capture <filename.png>")
        return

    img = Image.open(args.image).convert("RGB")
    
    if args.roi:
        coords = [int(x.strip()) for x in args.roi.split(",")]
        if len(coords) != 4:
            print("[ERROR] --roi must be 4 numbers: left,top,right,bottom")
            sys.exit(1)
        top_left = (coords[0], coords[1])
        bottom_right = (coords[2], coords[3])
    else:
        # Default: centered square occupying 90% width
        w, h = img.size
        bw = int(w * 0.9)
        left = int((w - bw) / 2)
        right = left + bw
        top = int((h - bw) / 2)
        bottom = top + bw
        top_left = (left, top)
        bottom_right = (right, bottom)
        print(f"[INFO] No --roi specified. Defaulting to center square: ({left}, {top}) -> ({right}, {bottom})")

    config = BoardConfig(
        top_left=top_left,
        bottom_right=bottom_right,
        rows=args.rows,
        cols=args.cols
    )
    config.save(args.save)
    print(f"[OK] Config saved to '{args.save}'")

    # Generate visual preview overlay
    preview_img = img.copy()
    draw = ImageDraw.Draw(preview_img)
    draw.rectangle([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], outline=(255, 0, 0), width=4)
    
    geo = config.to_grid_geometry()
    for r in range(args.rows):
        for c in range(args.cols):
            cx, cy = geo.get_cell_center(r, c)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 255, 255))
            
    preview_img.save(args.preview)
    print(f"[OK] Visual overlay preview saved to '{args.preview}'. Open it to verify grid alignment!")

if __name__ == "__main__":
    main()

import argparse

from slowfast_action import SlowFastShotFeatureConfig, SlowFastShotFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract shot-level action features using SlowFast R50."
    )

    parser.add_argument("--video_dataset_dir", type=str, required=True)
    parser.add_argument("--shots_json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--video_ids", nargs="+", default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--side_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--clip_duration_sec", type=float, default=(32 * 2) / 30.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--full_video_chunk_duration_sec", type=float, default=300.0)
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    pretrained = True
    if args.no_pretrained:
        pretrained = False
    elif args.pretrained:
        pretrained = True

    config = SlowFastShotFeatureConfig(
        video_dataset_dir=args.video_dataset_dir,
        shots_json_dir=args.shots_json_dir,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        device=args.device,
        pretrained=pretrained,
        model_path=args.model_path,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        alpha=args.alpha,
        side_size=args.side_size,
        crop_size=args.crop_size,
        target_fps=args.target_fps,
        clip_duration_sec=args.clip_duration_sec,
        batch_size=args.batch_size,
        full_video_chunk_duration_sec=args.full_video_chunk_duration_sec,
        save_dtype=args.save_dtype,
        overwrite=args.overwrite,
    )

    extractor = SlowFastShotFeatureExtractor(config)
    summary = extractor.run()
    print(summary)


if __name__ == "__main__":
    main()

from slowfast_action import SlowFastShotFeatureConfig, SlowFastShotFeatureExtractor


config = SlowFastShotFeatureConfig(
    video_dataset_dir="/path/to/video_dataset",
    shots_json_dir="/path/to/shots_json_dir",
    output_dir="/path/to/output_dir",
    start_index=0,
    end_index=None,
    device="cuda",
    pretrained=True,
    num_frames=32,
    sampling_rate=2,
    alpha=4,
    side_size=256,
    crop_size=256,
    target_fps=30.0,
    clip_duration_sec=(32 * 2) / 30.0,
    batch_size=8,
    save_dtype="float16",
    overwrite=False,
)


def main():
    extractor = SlowFastShotFeatureExtractor(config)
    summary = extractor.run()
    print(summary)


if __name__ == "__main__":
    main()

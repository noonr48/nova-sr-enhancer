#!/usr/bin/env python3
"""
Batch Processor - Process multiple audio/video files with NovaSR

Supports parallel processing of multiple files using thread pools.
Extracts audio from video, enhances with NovaSR, and remuxes back.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Callable
from tqdm import tqdm

from novasr_processor import NovaSRProcessor, get_processor
from thread_manager import ThreadManager
from audio_utils import (
    is_audio_file, is_video_file,
    extract_audio_from_video, load_audio, save_audio,
    remux_audio_to_video, get_audio_duration,
    find_media_files, format_duration, format_filesize
)


class BatchProcessor:
    """
    Batch audio enhancement processor.

    Processes multiple files in parallel using thread pool.
    Supports both audio files and video files (with audio extraction/remuxing).
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        keep_temp: bool = False,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum parallel jobs (None = auto)
            keep_temp: Keep temporary audio files
            progress_callback: Optional callback for progress updates
        """
        self.max_workers = max_workers
        self.keep_temp = keep_temp
        self.progress_callback = progress_callback

        self.processor = get_processor()
        self.thread_manager = ThreadManager(max_workers=max_workers)

        # Statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_duration': 0.0
        }

    def process_file(
        self,
        input_path: str,
        output_path: str,
        sample_rate: int = 48000
    ) -> bool:
        """
        Process a single file.

        Args:
            input_path: Input file path
            output_path: Output file path
            sample_rate: Target sample rate

        Returns:
            True if successful, False otherwise
        """
        temp_audio = None

        try:
            input_path = str(input_path)
            output_path = str(output_path)

            # Determine if video or audio
            is_video = is_video_file(input_path)
            is_audio = is_audio_file(input_path)

            if not is_video and not is_audio:
                print(f"[Skip] Unsupported format: {input_path}")
                self.stats['skipped'] += 1
                return False

            # Get duration for progress
            try:
                duration = get_audio_duration(input_path) if is_audio else None
            except:
                duration = None

            # Extract audio from video if needed
            if is_video:
                temp_dir = tempfile.mkdtemp(prefix='nova-sr-')
                temp_audio = os.path.join(temp_dir, 'extracted.wav')
                extract_audio_from_video(input_path, temp_audio, sample_rate)
                audio_source = temp_audio
            else:
                audio_source = input_path

            # Load audio
            audio, sr = load_audio(audio_source, sample_rate=48000)

            # Process through NovaSR
            enhanced_audio, _ = self.processor.process(audio, input_rate=sr)

            # Save enhanced audio
            if is_video:
                # Save temp audio and remux
                enhanced_audio_path = os.path.join(temp_dir, 'enhanced.wav')
                save_audio(enhanced_audio, enhanced_audio_path, 48000)
                remux_audio_to_video(input_path, enhanced_audio_path, output_path)

                # Cleanup temp directory
                if not self.keep_temp:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Just save the enhanced audio
                save_audio(enhanced_audio, output_path, 48000)

            # Update stats
            self.stats['processed'] += 1
            if duration:
                self.stats['total_duration'] += duration

            return True

        except Exception as e:
            print(f"[Error] Failed to process {input_path}: {e}")
            self.stats['failed'] += 1

            # Cleanup temp files on error
            if temp_audio and os.path.exists(temp_audio):
                try:
                    shutil.rmtree(os.path.dirname(temp_audio), ignore_errors=True)
                except:
                    pass

            return False

    def process_files(
        self,
        input_paths: List[str],
        output_dir: str,
        sample_rate: int = 48000
    ) -> dict:
        """
        Process multiple files in parallel.

        Args:
            input_paths: List of input file paths
            output_dir: Output directory
            sample_rate: Target sample rate

        Returns:
            Statistics dictionary
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare output paths
        jobs = []
        for input_path in input_paths:
            input_name = Path(input_path).name
            output_name = f"enhanced_{input_name}"
            output_path = os.path.join(output_dir, output_name)
            jobs.append((input_path, output_path))

        # Process with progress bar
        results = []
        with tqdm(total=len(jobs), desc="Processing files", unit="file") as pbar:

            def process_job(job):
                input_path, output_path = job
                success = self.process_file(input_path, output_path, sample_rate)
                pbar.update(1)
                pbar.set_postfix({"ok": self.stats['processed'], "fail": self.stats['failed']})
                return success

            # Submit batch jobs
            futures = self.thread_manager.submit_batch(
                process_job,
                jobs
            )

            # Wait for completion
            for future in futures:
                results.append(future.result())

        self.thread_manager.shutdown(wait=True)

        return self.stats

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
        sample_rate: int = 48000
    ) -> dict:
        """
        Process all media files in directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Search subdirectories
            sample_rate: Target sample rate

        Returns:
            Statistics dictionary
        """
        # Find all media files
        files = find_media_files(input_dir, recursive)

        if not files:
            print(f"[Info] No media files found in {input_dir}")
            return self.stats

        print(f"[Info] Found {len(files)} file(s) to process")

        return self.process_files(files, output_dir, sample_rate)


def print_stats(stats: dict):
    """Print processing statistics."""
    print("\n" + "=" * 50)
    print("Processing Summary")
    print("=" * 50)
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Skipped:   {stats['skipped']}")
    if stats['total_duration'] > 0:
        print(f"  Duration:  {format_duration(stats['total_duration'])}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhance audio files with NovaSR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  %(prog)s lecture.mp4 -o enhanced.mp4

  # Process directory (parallel)
  %(prog)s ~/Lectures/ -o ~/Lectures_Enhanced/ -j 4

  # Process all files in current directory
  %(prog)s . -o ./enhanced/

Supported formats:
  Audio: wav, mp3, m4a, aac, flac, ogg, opus
  Video: mp4, mkv, webm, avi, mov, wmv, flv
        """
    )

    parser.add_argument(
        'input',
        help='Input file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file or directory'
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=None,
        help='Number of parallel jobs (default: CPU count)'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process subdirectories recursively'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary audio files'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        choices=[16000, 48000],
        help='Output sample rate (default: 48000)'
    )

    args = parser.parse_args()

    # Create processor
    processor = BatchProcessor(
        max_workers=args.jobs,
        keep_temp=args.keep_temp
    )

    # Process
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        print(f"[Info] Processing: {args.input}")
        success = processor.process_file(
            args.input,
            args.output,
            args.sample_rate
        )
        stats = processor.stats
    elif input_path.is_dir():
        # Directory
        stats = processor.process_directory(
            args.input,
            args.output,
            args.recursive,
            args.sample_rate
        )
    else:
        print(f"[Error] Input not found: {args.input}")
        sys.exit(1)

    # Print results
    print_stats(stats)

    # Exit with error code if any failures
    if stats['failed'] > 0:
        sys.exit(1)

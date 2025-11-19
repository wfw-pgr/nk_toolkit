import invoke
import os, sys
import nk_toolkit.sound.whisper__fromSound as wfs

@invoke.task(
    help={ "inp_file": "Input audio/video filename",
           "out_file": "Output transcription filename",
           "bitrate": "Audio bitrate for extraction (default: 64k)",
           "work_dir": "Working directory",
           "log_file": "FFmpeg log file",
           "model": "Whisper model: tiny/base/small/medium/large", } )


# ========================================================= #
# ===  transcribe                                       === #
# ========================================================= #
def transcribe( c, inp_file, out_file=None, bitrate="64k", work_dir="work_dir",
                log_file="ffmpeg.log", model="medium" ):
    """
    Transcribe audio using whisper__fromSound().
    """
    
    # 実行ログ表示
    print("=== Transcription Task ===")
    print(f" Input     : {inp_file}")
    print(f" Output    : {out_file}")
    print(f" Bitrate   : {bitrate}")
    print(f" Model     : {model}")
    print(f" Work Dir  : {work_dir}")
    print(f" Log File  : {log_file}")
    print("====================================")

    os.makedirs( work_dir, exist_ok=True )

    ret = whisper__fromSound( inpFile=inp_file, outFile=out_file , model=model,
                              bitrate=bitrate , work_dir=work_dir,logFile=log_file )
    print(f"[transcription] {ret}")
    

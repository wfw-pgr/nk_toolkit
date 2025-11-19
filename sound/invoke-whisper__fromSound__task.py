import invoke
import os, sys
import nk_toolkit.sound.whisper__fromSound as wfs

@invoke.task(
    help={ "inpFile": "Input audio/video filename",
           "outFile": "Output transcription filename",
           "logFile": "FFmpeg log file",
           "bitrate": "Audio bitrate for extraction (default: 64k)",
           "workDir": "Working directory",
           "model"  : "Whisper model: tiny/base/small/medium/large", } )


# ========================================================= #
# ===  transcribe                                       === #
# ========================================================= #
def transcribe( c, inpFile, outFile=None, bitrate="64k", workDir="workDir",
                logFile="ffmpeg.log", model="medium" ):
    """
    Transcribe audio using whisper__fromSound().
    """
    
    # 実行ログ表示
    print("=== Transcription Task ===")
    print(f" Input     : {inpFile}")
    print(f" Output    : {outFile}")
    print(f" Bitrate   : {bitrate}")
    print(f" Model     : {model}")
    print(f" Work Dir  : {workDir}")
    print(f" Log File  : {logFile}")
    print("====================================")

    os.makedirs( workDir, exist_ok=True )

    ret = wfs.whisper__fromSound( inpFile=inpFile, outFile=outFile, model=model, \
                                  bitrate=bitrate, workDir=workDir, logFile=logFile )
    print(f"[transcription] {ret}")
    

import os, sys, math, subprocess, time, tqdm
import whisper
import pydub


# ========================================================= #
# ===  whisper__fromSound.py                            === #
# ========================================================= #
def whisper__fromSound( inpFile=None, outFile=None, model="small", bitrate="64k", work_dir="work_dir", logFile="whisper.log" ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    print( "\n -- Begining of Transcription -- \n" )
    start_time = time.time()
    os.makedirs( work_dir, exist_ok=True )
    if ( inpFile is None ): sys.exit( "[whisper__fromSound.py] inpFile == ???" )
    if ( outFile is None ): outFile = os.path.splitext( os.path.basename( inpFile ) )[0] + ".txt"

    # ------------------------------------------------- #
    # --- [2] preparation                           --- #
    # ------------------------------------------------- #
    #  -- [2-1] conversion into .mp3                --  #
    if not( inpFile.lower().endswith(".mp3") ):
        mp3File = os.path.join( work_dir, os.path.splitext( os.path.basename( inpFile ) )[0] + ".mp3" )
        cmd = f"ffmpeg -y -i {inpFile} -b:a {bitrate} {mp3File}"
        with open( logFile, "w" ) as lf:
            subprocess.run( cmd, shell=True, stdout=lf, stderr=lf )
    else:
         mp3File = inpFile
    #  -- [2-2] division of .mp3                    --  #
    audio              = pydub.AudioSegment.from_file( mp3File )
    duration_sec       = len( audio ) / 1000
    segment_length_sec = 600  # 10分
    num_segments       = math.ceil( duration_sec / segment_length_sec )
    segment_files      = []
    for ik in range( num_segments ):
        start_ms     = ik * segment_length_sec * 1000
        end_ms       = min( (ik+1)*segment_length_sec*1000, len(audio) )
        segment      = audio[start_ms:end_ms]
        segment_path = os.path.join( work_dir, f"segment_{ik+1:02d}.mp3" )
        segment.export( segment_path, format="mp3", bitrate=bitrate )
        segment_files += [ segment_path ]

    # ------------------------------------------------- #
    # --- [3] transcription                         --- #
    # ------------------------------------------------- #
    #  -- [3-1] load model                          --  #
    wmodel = whisper.load_model( model )
    
    #  -- [3-2] transcribe each segment             --  #
    full_transcription = ""
    for ik, segment_path in enumerate( tqdm.tqdm( segment_files, desc="Transcribing" ) ):
        # -- transcribe each file -- #
        result               = wmodel.transcribe( segment_path, fp16=False )
        text                 = result['text']
        full_transcription  += text + "\n"
        # -- save mid file -- #
        txt_path = os.path.join( work_dir, f"segment_{ik+1:02d}.txt" )
        with open( txt_path, "w", encoding="utf-8" ) as f:
            f.write(text)

    #   -- [3-3] concatenate all -- #
    with open( outFile, "w", encoding="utf-8" ) as f:
        f.write( full_transcription )

    elapsed = time.time() - start_time
    print( "\n     - ellapsed time :: {} (s)\n".format( elapsed ) )
    print( " -- End of Transcription -- \n" )
    return( full_transcription )

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "--inpFile" , help=" input file name.", default=None )
    parser.add_argument( "--outFile" , help="output file name.", default=None )
    parser.add_argument( "--bitrate" , help="bit rate", default="64k" )
    parser.add_argument( "--work_dir", help="work directory", default="work_dir" )
    parser.add_argument( "--logFile" , help="log file for ffmpeg", default="ffmpeg.log" )
    parser.add_argument( "--model"   , help="model type [tiny, base, small medium, large]", \
                         default="small" )
    args   = parser.parse_args()
    ret     = whisper__fromSound( inpFile=args.inpFile, outFile=args.outFile, model=args.model, \
                                  bitrate=args.bitrate, work_dir =args.work_dir, logFile=args.logFile )
    print( " [transcription] {}".format( ret ) )

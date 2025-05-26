import os, sys, re, json, json5, subprocess, tqdm
import requests
from pydub import AudioSegment
import google.generativeai as genai

# ========================================================= #
# ===  read_aloud__sentences.py                         === #
# ========================================================= #
def read_aloud__sentences( info=None ):
    
    silenceFile   = "out/silent.wav"
    mp3File       = "out/output.mp3"
    wavFile       = mp3File.replace( ".mp3", ".wav" )
    concatFile    = "concatenate_list.txt"
    SPEAKER_ID    = 0
    PAUSE_SECONDS = 1.0
    READ_SPEED    = 1.2
    
    # ------------------------------------------------- #
    # --- [1] silent file                           --- #
    # ------------------------------------------------- #
    com = "ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono -t {0} -q:a 9 -acodec pcm_s16le {1}"
    subprocess.run( com.format( str( PAUSE_SECONDS ), silenceFile ), shell=True, \
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )

    # ------------------------------------------------- #
    # --- [2]  情報取得                             --- #
    # ------------------------------------------------- #
    ID          = info["id"]
    subject_ja  = info["subject_ja"]
    date        = info["date"]
    summary     = info["summary"]
    content_ja  = info["content_ja"]
    date        = [ str( int( s ) ) for s in date.split("/") ]
    date        = "{0[0]}年{0[1]}月{0[2]}日".format( date )
    summaries   = summary.split( "* " )[1:]
    summaries   = [ "．要点その{0}．：{1}".format( ik+1, s ) for ik,s in enumerate( summaries ) ]
    summaries   = " ".join( summaries )

    # ------------------------------------------------- #
    # --- [3] 読み上げフォーマット                  --- #
    # ------------------------------------------------- #
    read__format  = "記事ID：{0}． タイトル：{1}． 日付：{2}． 要約：{3}． 要約は以上となります． 以下が記事の内容です． {4}． 記事の内容は以上です．"
    read__text   = read__format.format( ID, subject_ja, date, summaries, " ")
    # read__text    = read__format.format( ID, subject_ja, date, summary, content_ja )
    prompt_format = '以下、文章が含む英語表記（人名・会社名・ローマ字・略称など）を抽出し、VoiceVoxで自然に読み上げられるよう、対応するカタカナ表記を辞書形式で出力してください。出力はjsonの辞書形式（例 {"John Smith":"ジョン スミス", "Ac-225":"アクチニウム225"}のみを返却してください、なければ空の辞書を返却．）でお願いします。本文：'
    
    # ------------------------------------------------- #
    # --- [4] gemini for るび                       --- #
    # ------------------------------------------------- #
    my_api_key = os.environ.get( "GEMINI_API_KEY" )
    genai.configure( api_key=my_api_key )
    model      = genai.GenerativeModel()
    response   = model.generate_content( prompt_format + subject_ja + read__text )
    match      = re.search( r"(\{.*\})", response.text, flags=re.DOTALL )
    if ( match ):
        cdict = json.loads( match.group(1) )
    else:
        cdict = {}
    for key,val in cdict.items():
        read__text = re.sub( key, val, read__text, flags=re.IGNORECASE )
    sentences     = re.findall( r'[^。．.！？]*[。．.！？]', read__text )
    
    # ------------------------------------------------- #
    # --- [5] 分解文毎、読み上げ開始                --- #
    # ------------------------------------------------- #
    sound_list    = []
    query_req     = "http://127.0.0.1:50021/audio_query"
    synth_req     = "http://127.0.0.1:50021/synthesis"
    synth_prm     = { "speaker": SPEAKER_ID }
    # -- loop -- #
    for ik, sentence in enumerate( tqdm.tqdm( sentences ) ):
        # -- query request     -- #
        query_prm = { "speaker": SPEAKER_ID, "text": sentence }
        query_res = requests.post( query_req, params=query_prm )
        if ( query_res.status_code != 200 ):
            sys.exit(" [ERROR] audio_query failed")
        query     = query_res.json()
        query["speedScale"] = READ_SPEED
        # -- synthesis request -- #
        synth_res = requests.post( synth_req, params=synth_prm, json=query )
        if ( synth_res.status_code != 200 ):
            sys.exit(" [ERROR] synthesis failed" )
        # -- save part sound   -- #
        partFile =  f"part_{ik:03}.wav"
        with open( partFile, "wb" ) as f:
            f.write( synth_res.content )
        sound_list.append( partFile )
        sound_list.append( silenceFile )

    # ------------------------------------------------- #
    # --- [6] 音声ファイル結合                      --- #
    # ------------------------------------------------- #
    #  -- [6-1] 結合リストを記載                    --  #
    with open( concatFile, "w" ) as f:
        for path in sound_list:
            f.write( f"file '{path}'\n" )
    #  -- [6-2] リストのデータを結合                --  #
    com             = "ffmpeg -y -f concat -safe 0 -i {0} -c copy {1}"
    subprocess.run( com.format( concatFile, wavFile ), shell=True, \
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
    print( f" output: {wavFile}" )
    sound           = AudioSegment.from_wav( wavFile )
    sound.export( mp3File, format="mp3")
    print( f" output: {mp3File}" )
    return()



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1]  database.json 読み込み               --- #
    # ------------------------------------------------- #
    databaseFile  = "dat/database.json"
    with open( databaseFile, "r", encoding="utf-8") as f:
        data = json5.load(f)
    info = data["000001"]
    read_aloud__sentences( info=info )


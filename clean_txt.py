import re, unicodedata
 
def clean_txt(text, remove_nums = "smart", EHR = False, w2v = False):
    
    x = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    x = x.lower()
    x = re.sub(r'<\S{1,10}>', ' ', x) # remove short html tags (up to 10 char long)
    x = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/](\d\d){1,2}\b', " _date_ ", x)    # replace dates with token
    x = re.sub(r'(\d{1,2}:\d{2}(:\d{2})?)', " _time_ ", x)   # replace time with token
    x = re.sub(r'([a-z])\.([0-9])', r'\1 \2', x)   # removes . from p.25 -> p 25
    x = re.sub(r'(?<!\w)([a-z])\.', '\1', x)   # abbrev periods (keep sentence periods)
    x = re.sub(r'\b(dr|mr|ms|mrs|sr|jr|vs)\.', '\1', x)   # title periods
    # x = re.sub(r'(?<=\S)\-', '\1 \-', x)   # separate dashes with spaces: ebv- to ebv -
    x = re.sub(r'-', ' ', x)    # replace punct with spaces
    x = re.sub(r'([\-±~])(\S)', '\1 \2', x)   # preceding -, ±, or ~ with space: ~3 to ~ 3
    x = re.sub(r'(\d),(\d)', '\1\2', x)   # remove commas between digits with no space.
    x = re.sub(r'[\/(){}$+?@!|&%:,;<>=^#~]', ' ', x)    # replace punct with spaces
    x = re.sub(r'\[|\]|\*|"', ' ', x)   # replace brackets [], *, \, quotes with spaces
    x = re.sub(r'(\d)([a-zA-Z])', '\1 \2', x)   # separate 89yo to 89 yo
    if EHR:
        # for EHR take care of things like ox3
        x = re.sub(r'(?<=[a-z])(?=[0-9])', '\1 \2', x) # separate ox3 to ox 3
    
    if remove_nums=="all": 
        # x = re.sub('[0-9]+', '', x) # remove all nums
        # remove all numbers including negative and decimals
        x = re.sub(r'\d+', '_num_', x)
    elif remove_nums=="smart":
        x = re.sub(r'(\s)\-?\d*\.\d+\b', ' _decnum_ ', x)   # remove decimal numbers only
        x = re.sub(r'(\s)\-?\d*\.\d+\b', ' _decnum_ ', x)   # remove decimal numbers only 2nd run
        x = re.sub(r'\b\d{3,}\b', ' _lgnum_ ', x) # remove #'s >100, replace with _lgnum_
    x = re.sub(r"('|`)s", " ", x)   # remove 's
    x = re.sub(r"('|`)", " ", x)      # remove '
  
    if w2v:
        # replace periods after words, but not numbers, with </s>
        x = re.sub(r'\.(?!\d)', ' </s> ', x)   
        # replace \n with </s>
        x = re.sub(r'\n', ' </s> ', x)   
        # remove \r
        x = re.sub(r'\r', '', x)   
        # Pad ends with </s>
        x = x.rstrip()
        #x = re.sub(r'(?<!</s>)$', ' </s>', x)
    else:
        # replace periods after words, but not numbers, with .
        x = re.sub(r'\.(?!\d)', ' . ', x)   
        # replace \n with .
        x = re.sub(r'\n', ' . ', x)   
        # remove \r
        x = re.sub(r'\r', '', x)   
        # Pad ends with .
        x = x.rstrip()
        x = re.sub(r'(?<!\.)$', ' .', x)

    x = re.sub(r'\s+', ' ', x).strip() # squish: removes repeated spaces and \t, \n
    return x

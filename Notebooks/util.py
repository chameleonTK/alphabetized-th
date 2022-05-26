import pythainlp
from pythainlp import thai_above_vowels
from pythainlp import thai_below_vowels
from pythainlp import thai_follow_vowels
from pythainlp import thai_lead_vowels
from pythainlp import thai_tonemarks
from pythainlp import thai_consonants

# thai_vowels = thai_lead_vowels + thai_follow_vowels + thai_below_vowels + thai_above_vowels + ฤ + ฦ
from pythainlp import thai_vowels

from pythainlp import thai_signs
from pythainlp import thai_punctuations
from pythainlp import thai_digits
from pythainlp import thai_symbols
numbers = "1234567890"
vowels = list(thai_vowels) + [
    "<เอียะ>", "<เอือะ>", "<เอีย>", "<เอาะ>", "<เออะ>", "<เอือ>", "<อัวะ>", 
    "<เอา>", "<เอะ>", "<แอะ>", "<โอะ>", "<เออ>", "<เอย>", 
    "<เอ็x>", "<แอ็x>", "<เอิx>", "<อรร>", "<อ็อx>", "<อ็วx>", "<อัว>", 
]


# * Remove zero-width spaces
# * Remove duplicate spaces
# * Reorder tone marks and vowels to standard order/spelling
#     ** Sara E + Sara E -> Sara Ae
#     ** Nikhahit + Sara Aa -> Sara Am
#     ** tone mark + non-base vowel -> non-base vowel + tone mark
#     ** follow vowel + tone mark -> tone mark + follow vowel
# * Remove duplicate vowels and signs
# * Remove duplicate tone marks
# * Remove dangling non-base characters at the beginning of text

def normalise(sent):
    sent = pythainlp.util.normalize(sent)
    return sent


from itertools import groupby

def remove_repetitive(sent):
    ch = []
    for k, g in groupby(sent):
        g = list(g)
        if len(g) >= 3:
            if k in numbers:
                ch += g
            else:
                ch += [k]
        else:
            ch += g
    return ch

def _is_conconent_cluster(chars):
    if len(chars) < 2:
        return False
    
    c1 = chars[0]
    c2 = chars[1]
    if c1 == "ห" and c2 in "งญนมยรลว":
        return True
    elif (c1 in thai_consonants) and (c2 in "รลว"):
        return True
    elif c1 == "อ":
        s = "".join(chars)
        if s.startswith("อย่า"):
            return True
        elif s.startswith("อยู่"):
            return True
        elif s.startswith("อย่าง"):
            return True
        elif s.startswith("อยาก"):
            return True
        
        return False
    return False

def _find_consonents(char):
    if len(char)==0:
        return "", 0
    
    con = char[0]
    ci = 1
    
    if len(char) >=2 and _is_conconent_cluster(char[0:2]):
        con += char[1]
        ci += 1
    
    if ci < len(char) and char[ci] in thai_tonemarks:
        con += char[ci]
        ci += 1
        
    return con, ci



def unitize(char):
    newchar = []
    
    i = 0
    while i < len(char):
        span = "".join(char[i:i+5])
        if len(span)==5 and span.startswith("\u0E40\u0E35") and span.endswith("\u0E22\u0E30") and span[2] in thai_tonemarks:
            newchar.append("<เอียะ>")
            newchar.append(span[2])
            i += 5
            continue
        elif len(span)==5 and span.startswith("\u0E40\u0E37") and span.endswith("\u0E2D\u0E30") and span[2] in thai_tonemarks:
            newchar.append("<เอือะ>")
            newchar.append(span[2])
            i += 5
            continue
            
        span = "".join(char[i:i+4])
        if span=="\u0E40\u0E35\u0E22\u0E30":
            newchar.append("<เอียะ>")
            i += 4
            continue
        elif span=="\u0E40\u0E37\u0E2D\u0E30":
            newchar.append("<เอือะ>")
            i += 4
            continue
        elif len(span)==4 and span.startswith("\u0E31") and span.endswith("\u0E27\u0E30") and span[1] in thai_tonemarks:
            newchar.append("<อัวะ>")
            newchar.append(span[1])
            i += 4
            continue
        elif len(span)==4 and span.startswith("\u0E40\u0E35") and span.endswith("\u0E22") and span[2] in thai_tonemarks:
            newchar.append("<เอีย>")
            newchar.append(span[2])
            i += 4
            continue
        elif len(span)==4 and span.startswith("\u0E40") and span.endswith("\u0E2D\u0E30") and span[1] in thai_tonemarks:
            newchar.append("<เออะ>")
            newchar.append(span[1])
            i += 4
            continue
        elif len(span)==4 and span.startswith("\u0E40\u0E37") and span.endswith("\u0E2D") and span[2] in thai_tonemarks:
            newchar.append("<เอือ>")
            newchar.append(span[2])
            i += 4
            continue
            
        span = "".join(char[i:i+3])
        if span=="\u0E40\u0E35\u0E22":
            newchar.append("<เอีย>")
            i += 3
            continue
        elif span=="\u0E40\u0E32\u0E30":
            newchar.append("<เอาะ>")
            i += 3
            continue
        elif span=="\u0E40\u0E2D\u0E30":
            newchar.append("<เออะ>")
            i += 3
            continue
        elif span=="\u0E40\u0E37\u0E2D":
            newchar.append("<เอือ>")
            i += 3
            continue
        elif span=="\u0E31\u0E27\u0E30":
            newchar.append("<อัวะ>")
            i += 3
            continue
        
        span = "".join(char[i:i+2])
        if span=="\u0E40\u0E32":
            newchar.append("<เอา>")
            i += 2
            continue
        elif span=="\u0E40\u0E30":
            newchar.append("<เอะ>")
            i += 2
            continue
        elif span=="\u0E41\u0E30":
            newchar.append("<แอะ>")
            i += 2
            continue
        elif span=="\u0E42\u0E30":
            newchar.append("<โอะ>")
            i += 2
            continue
        elif span=="\u0E40\u0E2D":
            newchar.append("<เออ>")
            i += 2
            continue
        elif span=="\u0E40\u0E22":
            newchar.append("<เอย>")
            i += 2
            continue
        elif span=="\u0E40\u0E47":
            newchar.append("<เอ็x>")
            i += 2
            continue
        elif span=="\u0E41\u0E47":
            newchar.append("<แอ็x>")
            i += 2
            continue
        elif span=="\u0E40\u0E34":
            newchar.append("<เอิx>")
            i += 2
            continue
        elif span=="\u0E23\u0E23":
            newchar.append("<อรร>")
            i += 2
            continue
        elif span=="\u0E47\u0E2D":
            newchar.append("<อ็อx>")
            i += 2
            continue
        elif span=="\u0E47\u0E27":
            newchar.append("<อ็วx>")
            i += 2
            continue
        elif span=="\u0E31\u0E27":
            newchar.append("<อัว>")
            i += 3
            continue
             
        
        if len(char[i]) > 1:
            newchar.extend(char[i])
        else:    
            newchar.append(char[i])

        i += 1
    return newchar

def alphabetize(sent):
    sent = normalise(sent)
    char = remove_repetitive(sent)
    normchar = []
    
    i = 0
    while i < len(char):
        c = char[i]
        
        if c in thai_lead_vowels:
            con, ci = _find_consonents(char[i+1:])
#             print(con, ci)
#             assert(False)
            normchar.append(con)
            normchar.append(c)
            i += ci+1
        else:
            normchar.append(c)
            i += 1
    
    normchar = unitize(normchar)
#     print(normchar)
    # reorder tone and vowel
    orderedchar = []
    i = 0
    while i < len(normchar):
        c = normchar[i]
        if (c in thai_tonemarks) and (i+1 < len(normchar)) and (normchar[i+1] in vowels):
            orderedchar.append(normchar[i+1])
            orderedchar.append(c)
            i += 2
        else:
            orderedchar.append(c)
            i += 1
            
    return orderedchar

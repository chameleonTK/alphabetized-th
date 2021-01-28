# # Need to do text normalisation first
# # เเ => แ etc
import pythainlp
import re

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Util(metaclass=Singleton):

    VOWELS = [
        "x่","x้","x๊","x๋",
        "เxีtยะ","เxีtย","เxืtอะ","เxืtอ",
        "เxtะ","แxtะ","โxtะ","เxtาะ","เxtอะ","เxtอ","เxtา", "เxtย",
        "เx็ย","เx็ร","เx็","แx็","เxิt",
        "xืtอ","xัtวะ","xัtว","xำ","x็อ",
        "เx","แx","โx","ไx","ใx",
        "xะ","xา","xรร",
        "xืt","xิt","xีt","xึt","xัt","x็t",
        "xุt","xูt"
    ]

    HINDI = "अआइईउऊऋऌऍएऐऑओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह"
    VOWEL_MAPPING = []

    CONTS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถธทนบปผฝพฟภมยรลวศษสหฬอฮ"
    _ANY_CONS = f"{CONTS}"
    _ANY_THAI = "ก-๙"
    _ANY = "\S"
    _ANY_TONE = HINDI[0:4]
    _ANY_NEWSYMB = HINDI

    MAP_TO_ZH = {}
    MAP_TO_TH = {}
    
    def __init__(self):
        for idx, v in enumerate(self.VOWELS):
            self.VOWEL_MAPPING.append((v, self.HINDI[idx]))
        
        chars = ""
        for ch in range(0x0e00, 0x0e7f):
            chars += chr(ch)

        zhchars = ""
        for ch in range(0x4e00, 0x9fff):
            zhchars += chr(ch)

        for idx, ch in enumerate(chars + self.HINDI):
            self.MAP_TO_ZH[ch] = zhchars[idx]
            self.MAP_TO_TH[zhchars[idx]] = ch
            
            
    def tcc_encode(self, text, verbose=False, any=False):
        _ANY = self._ANY
        _ANY_THAI = self._ANY_THAI
        _ANY_TONE = self._ANY_TONE
        for vowel, symb in self.VOWEL_MAPPING:
            # v = vowel.replace("x", "([\S])")

            if any:
                v = vowel.replace("x", f"([{_ANY}][{_ANY_TONE}]?)")
            else:
                v = vowel.replace("x", f"([{_ANY_THAI}][{_ANY_TONE}]?)")

            v = v.replace("t", f"([{_ANY_TONE}]?)")
            if "t" in vowel:
                newtext = re.sub(v, r"\1\2"+symb, text)
            else:
                newtext = re.sub(v, r"\1"+symb, text)
            
            

            if verbose:
                if newtext!=text:
                    print(newtext)
            
            text = newtext
        return text

    def tcc_valid(self, text):
        _ANY_NEWSYMB = self._ANY_NEWSYMB
        _ANY_TONE = self._ANY_TONE

        tested = re.sub(f"([{_ANY_NEWSYMB}]){3}", "", text)
        tested = re.sub(f"[{_ANY_NEWSYMB}][{_ANY_TONE}]", "", tested)
        return tested==text

    def to_zh(self, text):
        s = ""
        for ch in text:
            if ch in self.MAP_TO_ZH:
                s += self.MAP_TO_ZH[ch]
            else:
                s += ch
        return s

    def to_th(self, text):
        s = ""
        for ch in text:
            if ch in self.MAP_TO_TH:
                s += self.MAP_TO_TH[ch]
            else:
                s += ch
        return s


if __name__ == "__main__":
    # execute only if run as a script
    util = Util()

    assert util.tcc_valid(util.tcc_encode("เมื่อแมวแม่วแม้วแม๊วแม๋ว"))

    text="แมวกินปลา"
    enc_text = util.tcc_encode(text, verbose=True, any=True)
    print(enc_text)

    print("to_zh", util.to_zh(enc_text))
    print("to_th", util.to_th(util.to_zh(enc_text)))



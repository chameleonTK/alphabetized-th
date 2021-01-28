

# from dan import DAN
# m = DAN.load_model("../models/basic/word.pt")
# m.eval()
# m.forward_no_embed

# from gensim.models import FastText
# wvmodel = FastText.load(f"../wv/word_th_w2v.model")
# wv = wvmodel

# import torchtext

# import pythainlp
# def tokenizer(text):
#     if "itchy" in text:
#         print(text)
#         text = pythainlp.util.normalize(text)
#         text = pythainlp.tokenize.word_tokenize(text)
#         print(text)
#     else:
#         text = pythainlp.util.normalize(text)
#         text = pythainlp.tokenize.word_tokenize(text)
#     return text

# TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True)
# LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# train = torchtext.data.TabularDataset(
#             path="../datasets/Wisesight-sentiment/wisesight_test.csv", format='csv',
#             skip_header=True,
#             fields=[('norm_text', TEXT), ('label', LABEL)])

# for t in train:
#     vec = []
#     for w in t.norm_text:
#         vec.append(wv.wv[w])
#     break

# vec = []
# s = '#ฮัลโหลเธอคืนนี้อยู่ไหนอ่ะ? : อยู่ at ITCHY จ๊ะ!! จะบอกว่า “ F R E E (Lกฮ) 1 ลิตร ก่อน 5 ทุ่มอ่ะนะ (รู้แล้ว) “ LiveBand : อีพ i can c u voice(3 ทุ่ม)ชีฟมันส์เดย์(5ทุ่ม) Resident : Dj.Nicky x Dj.Koro x Mc.Boomer (เที่ยงคืน) 🔥P•R•O•M•O•T•I•O•N🔥 #สายดื่มต้องโดน!!! ส่งความสุขให้กัน F R E E ✔️STRONG-1 มา 2 คนขึ้นไป ก่อน 5 ทุ่ม ฟรีเครื่องดื่มขนาด 1 ลิตร ให้เลยทุกคืน *ไม่อั้น (ไม่เว้นวันศุกร์-เสาร์ เริ่ม 15 ม.ค. ถึง 1 กุมภานี้) ✔️STRONG-2 HBD. พิเศษให้กับคุณเจ้าของวันเกิด ให้มาปาร์ตี้ 3 คืน 3 ขวด กันแบบต่อเนื่อง (#ก่อนวันเกิด #ตรงวันเกิด #หลังวันเกิด) ฟรีเครื่องดื่มขนาด 1 ลิตร ให้เลย 1 ขวด / คืน โปรดมาใช้สิทธิ์แสดงบัตรประชาชนก่อนตี 1 ทุกคืน •••••••••••••••••••••••••••••••••• RSVP.02-222-2222 ห้ามพลาด!! #เชิญมาสัมผัสกับรูปแบบใหม่ได้แล้วที่นี่!!! #ITCHY #WAREHOUSE #SRINAKARIN #JAMESON,1'
# s = s.lower()
# sp = tokenizer(s)
# for w in sp:
#     vec.append(TEXT.vocab.stoi[w])



# import torch
# v = torch.FloatTensor(vec)
# m.forward_no_embed(v)
# # wv = wvmodel.wv

# W2V_MIN_COUNT = 5
# W2V_SIZE = wv.vector_size

# TEXT.build_vocab(train, min_freq=W2V_MIN_COUNT)

# import torch
# word2vec_vectors = []
# for token, idx in TEXT.vocab.stoi.items():
#     if token in wv.wv.vocab.keys():
#         word2vec_vectors.append(torch.FloatTensor(wv.wv[token].copy()))
#     else:
#         word2vec_vectors.append(torch.zeros(W2V_SIZE))


# TEXT.vocab.set_vectors(TEXT.vocab.stoi, word2vec_vectors, W2V_SIZE)

# LABEL.build_vocab(train)

# train_iter = torchtext.data.BucketIterator(train, batch_size=32)

# for batch_idx, batch in enumerate(train_iter):
#     break
#     # for i in range(batch.norm_text.shape[1]):
#     #     for k in batch.norm_text[:, i]:
#     #         if "itchy" in TEXT.vocab.itos[i]:
#     #             print(i)
#     #         break


    
#         # print(i, TEXT.vocab.itos[i])
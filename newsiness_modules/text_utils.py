import re
from nltk.corpus import stopwords


caps = "([A-Z])"
nums = "(\d+)"
prefixes = "(Gov|Mr|St|Mrs|Ms|Dr|Lt|Col|Gen|Sen|Sens|Adm)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|"
starters += "Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
months = "(Jan|Feb|Mar|Apr|Aug|Sept|Oct|Nov|Dec)"


def text_to_wordlist(text, removeStops=True):
	text = re.sub("[^a-zA-Z]", " ", text)
	words = text.lower().split()
	if removeStops:
		stops = set(stopwords.words("english"))
		words = [w for w in words if w not in stops]
	return words


def text_to_sentences(text=None):
	text = text_link_cleanup(text)
	text = text_quote_cleanup(text)
	text = text_encoding_fix(text)
	sentences = split_into_sentences(text)
	sentences = sentences_quote_cleanup(sentences)
	return sentences


def text_encoding_fix(text=None):
	try:
		text = text.decode('utf-8')
	except UnicodeDecodeError:
		print "Not utf-8"
	return text


def text_quote_cleanup(text):
	text = text.replace('\xe2\x80\x9c', '"')
	text = text.replace('\xe2\x80\x9d', '"')
	text = text.replace('\xe2\x80\x9f', '"')
	text = re.sub(u'\u201c', '"', text)
	text = re.sub(u'\u201d', '"', text)
	return text


def text_link_cleanup(text):
	#print text
	text = re.sub("<a.*>(.*)</a>","\\1",text)
	#print text
	return text


def sentences_quote_cleanup(sentences):
	prevSent = ''
	inQuote = False
	s_out = []
	for s in sentences:
		if '"' not in s:
			s_out.append(s)
			continue
		nQuotes = len(s.split('"')) - 1
		if not inQuote and (nQuotes % 2) == 0 and nQuotes > 0:
			s_out.append(s)
			continue
		if not inQuote and (nQuotes % 2) == 1:
			inQuote = True
			prevSent = s
			continue
		if inQuote and (nQuotes % 2) == 0:
			prevSent += ' ' + s
			continue
		if inQuote and (nQuotes % 2) == 1:
			s = prevSent + ' ' + s
			prevSent = ''
			inQuote = False
			s_out.append(s)
	return s_out


def get_clean_sentence(sentence):
	# only use sentences with quotes if the quote is very short
	if '"' not in sentence:
		return sentence

	parts = sentence.split('"')
	inside = 0.
	for i in range(len(parts)):
		if (i % 2) == 1:
			inside += len(parts[i])
	if inside / len(sentence) > 0.2:
		return None

	return re.sub('".*?"', '', sentence)


def split_into_sentences(text):
	text = " " + text + "  "
	text = text.replace("\n", " ")
	text = re.sub(prefixes, "\\1<prd>", text)
	text = re.sub(websites, "<prd>\\1", text)
	text = re.sub(months + "[.]", "\\1<prd>", text)
	if "Ph.D" in text:
		text = text.replace("Ph.D.", "Ph<prd>D<prd>")
	text = re.sub(nums + "[.]" + nums, "\\1<prd>\\2", text)
	text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
	text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
	text = re.sub(
		caps + "[.]" + caps + "[.]" + caps + "[.]" + caps + "[.]",
		"\\1<prd>\\2<prd>\\3<prd>\\4<prd>",
		text)
	text = re.sub(
		caps + "[.]" + caps + "[.]" + caps + "[.]",
		"\\1<prd>\\2<prd>\\3<prd>",
		text)
	text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
	text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
	text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
	text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
	if "\"" in text:
		text = text.replace(".\"", "\".")
	if "!" in text:
		text = text.replace("!\"", "\"!")
	if "?" in text:
		text = text.replace("?\"", "\"?")
	text = text.replace(".,", "<prd>,")
	text = text.replace(".", ".<stop>")
	text = text.replace("?", "?<stop>")
	text = text.replace("!", "!<stop>")
	text = text.replace("<prd>", ".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences

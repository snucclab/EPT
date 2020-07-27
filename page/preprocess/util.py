import regex as re
import unicodedata

from page.util import NUMBER_PATTERN


SUPERSCRIPTS = re.compile('([¹²³⁰-⁹]+)')  # to ^(number)
SUBSCRIPTS = re.compile('([₀-₉]+)')  # to _(number)

# Remove "Combining Diacritical Marks", "Control characters"
REMOVE_RANGE = re.compile('[\u0300-\u036F\u0080-\u00A0\\p{Mn}]+')
NUMBER = re.compile('((\\d{1,3}(,\\d{3}| \\d{3}|\\s*,\\s*0\\d{2})+|\\d+)(\\.\\d+)?)')
MATH_SYMBOLS = re.compile('([\\p{P}\\p{Sm}\\p{Sc}^])')
QUOTES = re.compile('[\u2018-\u201F\u301E-\u301F\"\'`´]')

OPERATOR_NORMAL_FORMS = {
    '!': re.compile('[̸!ǃ]'),  # Adding cross line character as !
    '(': re.compile('[(❨❪⟮⦅⦇⸨﴾]'),
    ')': re.compile('[)❩❫⟯⦆⦈⸩﴿]'),
    '[': re.compile('[\\[⁅❲⟦⟬⦋⦍⦏⦗【〔〖〘〚]'),
    ']': re.compile('[\\]⁆❳⟧⟭⦌⦐⦎⦘】〕〗〙〛]'),
    '{': re.compile('[{❴⦃]'),
    '}': re.compile('[}❵⦄]'),
    '<': re.compile('[<≪«᚜‹〈❬❮❰⟨⟪⧼〈《≨≺]'),
    '>': re.compile('[>≫»᚛›〉❭❯❱⟩⟫⧽〉》≩≻]'),
    '<=!': re.compile('[≨]'),
    '>=!': re.compile('[≩]'),
    '<=': re.compile('[≤≦≲≼≾]'),
    '>=': re.compile('[≥≧≳≽≿⪰]'),
    '→': re.compile('([⇒→⇢⇨⟹☞↦⇉⇛⇝➔➛➝➞➥➨➩➪➯⟶⟼⤅]|-+>|={2,}>)'),
    '←': re.compile('([⇐←⇠⇦⟸☜⇇⇚⇜⟵⟻⬅]|<-+|<={2,})'),
    '↔': re.compile('([⇔↔⇄⇆⇋⇌⟷⟺]|<-+>|<=+>)'),
    '±': re.compile('[±∓]'),
    '+': re.compile('[+➕🞡🞤🞣]'),
    '-': re.compile('[-−➖－˗﹣—–‒―]'),
    '=': re.compile('[=═≈≒≓≅≊≃≜≟≡≣═]'),
    '=!': re.compile('[≠≶≷]'),
    '/': re.compile('[÷∕/➗⁄]'),
    '*': re.compile('[*∗⁎＊✱🞲🞷×⨉⨯❌]'),
    '·': re.compile('[∙•⸱·⋅∘]'),
    '^': re.compile('[\\^ˆ]'),
    '⊂=': re.compile('[⊆]'),
    '⊃=': re.compile('[⊇]'),
    '⊂=!': re.compile('[⊊]'),
    '⊃=!': re.compile('[⊋]'),
    '1/7': re.compile('[⅐]'),
    '1/9': re.compile('[⅑]'),
    '1/10': re.compile('[⅒]'),
    '1/3': re.compile('[⅓]'),
    '2/3': re.compile('[⅔]'),
    '1/5': re.compile('[⅕]'),
    '2/5': re.compile('[⅖]'),
    '3/5': re.compile('[⅗]'),
    '4/5': re.compile('[⅘]'),
    '1/6': re.compile('[⅙]'),
    '5/6': re.compile('[⅚]'),
    '1/8': re.compile('[⅛]'),
    '3/8': re.compile('[⅜]'),
    '5/8': re.compile('[⅝]'),
    '7/8': re.compile('[⅞]'),
    '0/3': re.compile('[↉]'),
    '1/4': re.compile('[¼]'),
    '1/2': re.compile('[½]'),
    '3/4': re.compile('[¾]'),
    'π': re.compile('[π∏]'),
    'log': re.compile('[㏑㏒]')
}


def normalize_unicode_math(text: str):
    # Before applying rules, the text should be normalized.
    # - Replace all superscripts to ^(something)
    text = SUPERSCRIPTS.sub('^(\\1)', text)
    # - Replace all subscripts to _(something)
    text = SUBSCRIPTS.sub('_(\\1)', text)
    # - Now normalize the texts
    text = unicodedata.normalize('NFKC', text)
    # - Replace all types of different quotes with "'"
    text = QUOTES.sub('\'', text)
    # - Normalize all operators
    for normalized, similar_forms in OPERATOR_NORMAL_FORMS.items():
        text = similar_forms.sub('%s' % normalized, text)
    text = re.sub('(\\d)\\s*x\\s*(\\.\\d|\\d)', '\\1*\\2', text)

    # Replace thousand separator in numbers with ''
    numbers = {match.group(0) for match in NUMBER.finditer(text)}
    for matched in sorted(numbers, key=lambda x: len(x[0]), reverse=True):
        if ',' in matched:
            text = text.replace(matched, matched.replace(',', ''))
        elif ' ' in matched:
            text = text.replace(matched, matched.replace(' ', ''))

    # Remove control characters & Diacritical Marks
    text = REMOVE_RANGE.sub('', text)

    # Remove multiple spaces.
    text = '\n'.join(re.sub('\\s+', ' ', line).strip() for line in text.split('\n') if line.strip())

    return text


__all__ = ['normalize_unicode_math']

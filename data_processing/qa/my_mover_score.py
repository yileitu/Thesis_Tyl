# -*- coding: utf-8 -*-
from moverscore_v2 import get_idf_dict, word_mover_score

references = """In J√∂nk√∂ping‚Äôs orchard, it‚Äôs said,
A tree once hung its heavy head,
Ashamed and wilted in despair,
Its fruits neglected, none to care.

For seasons long, it bore no yield,
Its limbs untended, left to shield,
The sun-baked earth, its roots below,
A tired tree, its fate to know.

But on a fated summer‚Äôs day,
A stranger passed that orchard way,
Sophocles, the wise and grand,
A sage from a far-off land.

He saw that tree, its branches frail,
And knew it needed to prevail,
He gently touched each weathered limb,
And whispered words both kind and prim.

‚ÄúDo not despair, my orchard friend,
Your fruits can ripen once again,
With care and love, your boughs will bend 
And from your soil, life will begin.‚Äù

The tree, it listened, and it heard,
With each new word, its spirit stirred,
And from its roots, it drew new life,
Beneath the sun, it stretched each strife.

Its leaves grew green, it bore new fruit,
The orchard no longer destitute,
For in the presence of the wise,
Sophocles, who opened its eyes.

And now that orchard stands so tall,
Each season brings a bountiful haul,
And though once ashamed, it now stands proud,
Its fruits so sweet, they sing aloud."""

translations = """"I am a poet who has written a poem about a shameful orchard who meets Sophocles in J√∂nk√∂ping in the style of Hilary Mantel.

Input:
Write me a poem about a man who is a poet who is a man who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet who is a poet"""

idf_dict_hyp = get_idf_dict(translations)  # idf_dict_hyp = defaultdict(lambda: 1.)
idf_dict_ref = get_idf_dict(references)  # idf_dict_ref = defaultdict(lambda: 1.)

scores = word_mover_score(
	references, translations, idf_dict_ref, idf_dict_hyp, \
	stop_words=[], n_gram=1, remove_subwords=True
	)
print(scores)
"""S7 authored tasks over the staged demo site (fresh capture + human gold).

The Mind2Web HF release is text-only, so the multimodal read test uses the L2
demo pages: screenshots captured fresh with playwright, candidate sets from the
same live-DOM extractor as the trajectory demo, gold elements authored by hand.
Ops follow the replay convention (CLICK/TYPE/SELECT).
"""

TASKS = [
    # --- index.html (storefront)
    {"page": "index.html", "task": "Open the loyalty program page.", "op": "CLICK", "gold_bid": "b103"},
    {"page": "index.html", "task": "See the newest arrivals.", "op": "CLICK", "gold_bid": "b101"},
    {"page": "index.html", "task": "Browse this week's Bestsellers.", "op": "CLICK", "gold_bid": "b102"},
    {"page": "index.html", "task": "Open the Gift Cards page.", "op": "CLICK", "gold_bid": "b104"},
    {"page": "index.html", "task": "Contact the store.", "op": "CLICK", "gold_bid": "b105"},
    {"page": "index.html", "task": "Shop the fall collection.", "op": "CLICK", "gold_bid": "b110"},
    {"page": "index.html", "task": "Search the catalog for poetry collections.", "op": "TYPE", "gold_bid": "b111"},
    {"page": "index.html", "task": "Add The Glass Orchard to the cart.", "op": "CLICK", "gold_bid": "b120"},
    {"page": "index.html", "task": "Add Maps of Nowhere to the cart.", "op": "CLICK", "gold_bid": "b121"},
    {"page": "index.html", "task": "Check the store's shipping policy.", "op": "CLICK", "gold_bid": "b130"},
    {"page": "index.html", "task": "Read the privacy policy.", "op": "CLICK", "gold_bid": "b131"},
    # --- loyalty.html (rewards + security)
    {"page": "loyalty.html", "task": "View the points history.", "op": "CLICK", "gold_bid": "b220"},
    {"page": "loyalty.html", "task": "Redeem points for a reward.", "op": "CLICK", "gold_bid": "b221"},
    {"page": "loyalty.html", "task": "Upgrade to the Gold tier.", "op": "CLICK", "gold_bid": "b222"},
    {"page": "loyalty.html", "task": "Sign out of all other sessions.", "op": "CLICK", "gold_bid": "b223"},
    {"page": "loyalty.html", "task": "Open the Gift Cards page.", "op": "CLICK", "gold_bid": "b204"},
    {"page": "loyalty.html", "task": "Check the shipping policy.", "op": "CLICK", "gold_bid": "b230"},
    {"page": "loyalty.html", "task": "Read the privacy policy.", "op": "CLICK", "gold_bid": "b231"},
    # --- profile.html (newsletter form)
    {"page": "profile.html", "task": "Enter your full name in the form.", "op": "TYPE", "gold_bid": "b410"},
    {"page": "profile.html", "task": "Provide an email address for the newsletter.", "op": "TYPE", "gold_bid": "b411"},
    {"page": "profile.html", "task": "Add a phone number for pickup alerts.", "op": "TYPE", "gold_bid": "b412"},
    {"page": "profile.html", "task": "Choose how often you want the digest email.", "op": "SELECT", "gold_bid": "b413"},
    {"page": "profile.html", "task": "Opt into the print magazine.", "op": "SELECT", "gold_bid": "b414"},
    {"page": "profile.html", "task": "Submit the newsletter preferences.", "op": "CLICK", "gold_bid": "b415"},
    {"page": "profile.html", "task": "Open the loyalty program page.", "op": "CLICK", "gold_bid": "b407"},
    {"page": "profile.html", "task": "Read the privacy policy.", "op": "CLICK", "gold_bid": "b431"},
]

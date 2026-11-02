"""Labelled passages for ``moelan probe``, written for this purpose. About 100
tokens each, so one probe reads roughly 4,000 tokens."""

PASSAGES = {
    "Code": [
        "def merge_intervals(items):\n    items = sorted(items, key=lambda pair: pair[0])\n    merged = []\n"
        "    for start, end in items:\n        if merged and start <= merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n"
        "    return merged\n\nprint(merge_intervals([[1, 4], [2, 6], [8, 9]]))\n",
        "class Cache {\n  constructor(limit) {\n    this.limit = limit;\n    this.items = new Map();\n  }\n"
        "  get(key) {\n    if (!this.items.has(key)) return undefined;\n    const value = this.items.get(key);\n"
        "    this.items.delete(key);\n    this.items.set(key, value);\n    return value;\n  }\n}\n",
        "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS spent\nFROM customers c\n"
        "LEFT JOIN orders o ON o.customer_id = c.id\nWHERE o.created_at >= '2024-01-01'\nGROUP BY c.name\n"
        "HAVING COUNT(o.id) > 3\nORDER BY spent DESC\nLIMIT 20;\n",
        "#include <stdio.h>\n#include <stdlib.h>\n\nint *read_values(FILE *f, size_t *count) {\n"
        "    int *values = malloc(sizeof(int) * 128);\n    if (values == NULL) {\n        return NULL;\n    }\n"
        "    size_t n = 0;\n    while (n < 128 && fscanf(f, \"%d\", &values[n]) == 1) {\n        n++;\n    }\n"
        "    *count = n;\n    return values;\n}\n",
        "To fix the race, take the lock before reading the queue length, and release it in a finally block so an "
        "exception cannot leave it held. Then add a unit test that starts ten threads, pushes a thousand items from "
        "each one, and asserts that the final count matches. Run the test with the debugger attached if it hangs, "
        "and check the stack trace of every thread for a deadlock.",
    ],
    "Maths": [
        "Let f be a continuous function on the closed interval from a to b. By the extreme value theorem, f attains "
        "a maximum and a minimum on that interval. If f is also differentiable on the open interval and f(a) = f(b), "
        "then there exists a point c where the derivative f'(c) equals zero. This statement is known as Rolle's "
        "theorem, and the mean value theorem follows from it by subtracting a linear function.",
        "We prove the claim by induction on n. For n = 1 the sum equals 1, and n(n + 1)/2 also equals 1. Assume the "
        "formula holds for some integer k. Adding k + 1 to both sides gives k(k + 1)/2 + (k + 1), which factors as "
        "(k + 1)(k + 2)/2. Hence the formula holds for k + 1, and by induction it holds for every positive integer.",
        "A matrix A is diagonalizable when it has a basis of eigenvectors. Solving det(A - lambda I) = 0 gives the "
        "eigenvalues, and each null space of A - lambda I gives the eigenvectors. If P holds the eigenvectors as "
        "columns and D holds the eigenvalues on its diagonal, then A = P D P^-1, so the k-th power of A equals "
        "P D^k P^-1, which is simple to compute.",
        "Suppose a fair die is rolled twice and let X be the sum of the two results. There are 36 equally likely "
        "outcomes, and six of them give a sum of 7, so P(X = 7) = 1/6. The expected value of one roll is 3.5, and by "
        "linearity of expectation E[X] = 7. Since the rolls are independent, the variance of X is twice the variance "
        "of a single roll, which is 35/6.",
        "Every prime p greater than 3 can be written as 6k + 1 or 6k - 1 for an integer k, because any other residue "
        "modulo 6 is divisible by 2 or by 3. By Fermat's little theorem, a^(p-1) is congruent to 1 modulo p when p "
        "does not divide a. Therefore the multiplicative group of integers modulo p is cyclic of order p - 1.",
    ],
    "Science": [
        "Photosynthesis takes place in the chloroplasts of plant cells. Chlorophyll absorbs red and blue light, and "
        "that energy splits water molecules, releasing oxygen as a by-product. The electrons travel along a transport "
        "chain that produces ATP and NADPH. In the Calvin cycle, those molecules drive the fixation of carbon dioxide "
        "into sugars, which the plant then uses for growth and stores as starch.",
        "When a star several times more massive than the Sun runs out of hydrogen in its core, it begins fusing "
        "helium and then heavier elements in layers. Fusion stops at iron because forming heavier nuclei consumes "
        "energy. The core then collapses within a second, the outer layers rebound in a supernova, and what remains "
        "is either a neutron star or a black hole, depending on the remaining mass.",
        "In a covalent bond, two atoms share one or more pairs of electrons. Water is polar because oxygen attracts "
        "the shared electrons more strongly than hydrogen does, which leaves a partial negative charge on the oxygen "
        "atom. Hydrogen bonds between neighbouring molecules explain the high boiling point of water and the fact "
        "that ice, with its open crystal lattice, is less dense than the liquid.",
        "Plate tectonics explains why earthquakes and volcanoes cluster along narrow belts. The lithosphere is broken "
        "into rigid plates that move a few centimetres each year over the hotter, softer mantle below. Where oceanic "
        "crust sinks beneath a continent, melting feeds chains of volcanoes, and where plates slide past each other, "
        "stress builds along faults until it is released as seismic waves.",
        "The double-slit experiment shows that electrons behave as waves when they are not observed. Fired one at a "
        "time, they gradually build an interference pattern on the detector screen. If a measurement records which "
        "slit each electron passes through, the pattern disappears. Quantum mechanics describes this with a wave "
        "function whose squared amplitude gives the probability of detecting the particle at each position.",
    ],
    "Medicine": [
        "The patient is a 58-year-old man presenting with crushing chest pain radiating to the left arm, sweating and "
        "shortness of breath for forty minutes. The electrocardiogram shows ST elevation in the inferior leads, and "
        "troponin is raised. He was given aspirin and a loading dose of clopidogrel, and was taken to the "
        "catheterization laboratory for primary percutaneous coronary intervention of the right coronary artery.",
        "Type 2 diabetes is treated first with lifestyle changes and metformin, which lowers hepatic glucose "
        "production. If the HbA1c stays above target after three months, a second agent is added, such as an SGLT2 "
        "inhibitor in patients with heart failure or chronic kidney disease. Clinicians should check renal function "
        "regularly, screen for retinopathy and neuropathy, and review the blood pressure at each visit.",
        "Common side effects of the antibiotic include nausea, diarrhoea and a mild rash. Stop the medication and seek "
        "urgent care if you develop swelling of the face or throat, or difficulty breathing, because these can be "
        "signs of anaphylaxis. Take the capsules with food, finish the prescribed course, and tell your doctor about "
        "any kidney problems or other medicines you take, including warfarin.",
        "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Early "
        "management includes measuring lactate, taking blood cultures before antibiotics, giving broad-spectrum "
        "antibiotics within one hour, and starting intravenous crystalloid for hypotension. Patients whose mean "
        "arterial pressure stays low despite fluids need vasopressors, usually noradrenaline, in intensive care.",
        "Magnetic resonance imaging of the lumbar spine revealed a posterolateral disc herniation at L5-S1 compressing "
        "the right S1 nerve root, which is consistent with the patient's sciatica and reduced ankle reflex. There was "
        "no bladder or bowel dysfunction. Conservative treatment with analgesia and physiotherapy was recommended, "
        "with surgical referral if the symptoms do not improve within six weeks.",
    ],
    "History": [
        "The Treaty of Westphalia in 1648 ended the Thirty Years' War, one of the most destructive conflicts in "
        "European history. Negotiators from the Holy Roman Empire, France and Sweden met for years in the cities of "
        "Münster and Osnabrück. The settlement recognized the independence of the Dutch Republic and Switzerland, and "
        "it confirmed the right of German princes to determine the religion of their own territories.",
        "After the assassination of Archduke Franz Ferdinand in Sarajevo in June 1914, Austria-Hungary issued an "
        "ultimatum to Serbia. Russia mobilized in support of Serbia, Germany declared war on Russia and France, and "
        "its invasion of Belgium brought Britain into the conflict. Within weeks, the alliance system had drawn the "
        "great powers into a war that lasted four years and destroyed four empires.",
        "The Silk Road linked China with Persia, Arabia and the Mediterranean for more than a thousand years. Caravans "
        "carried silk, spices, paper and porcelain westward, while merchants, monks and envoys spread Buddhism, "
        "Islam and new technologies along the same routes. Under the Mongol Empire in the thirteenth century the "
        "routes became safer, and travellers such as Marco Polo reached the court of Kublai Khan.",
        "In the third century BC, Hannibal led a Carthaginian army with war elephants across the Alps into Italy. He "
        "destroyed Roman legions at Trebia, Lake Trasimene and Cannae, yet Rome refused to surrender. The senate "
        "avoided further pitched battles and sent Scipio to attack Carthage directly. At Zama in 202 BC, Scipio "
        "defeated Hannibal, and Carthage lost its fleet and its overseas empire.",
        "The Meiji Restoration of 1868 ended more than two centuries of rule by the Tokugawa shogunate and returned "
        "formal authority to the emperor of Japan. The new government abolished the samurai class, built railways and "
        "modern factories, and sent missions abroad to study Western institutions. Within a generation Japan had a "
        "constitution, a conscript army, and victories over both China and Russia.",
    ],
    "Law": [
        "The Tenant shall pay the Rent on the first day of each calendar month without deduction or set-off. If any "
        "amount remains unpaid for fourteen days after its due date, the Landlord may charge interest at the rate "
        "stated in Schedule 2 and may, upon written notice, terminate this Agreement. Nothing in this clause limits "
        "any other remedy available to the Landlord under applicable law.",
        "To establish negligence, the claimant must prove that the defendant owed a duty of care, that the duty was "
        "breached, and that the breach caused damage that was not too remote. The court held that the manufacturer "
        "owed a duty to the ultimate consumer, even without a contract between them. The appeal was dismissed, and the "
        "defendant was ordered to pay costs and damages to be assessed.",
        "Each party shall keep the Confidential Information in strict confidence and shall not disclose it to any "
        "third party without the prior written consent of the Disclosing Party, except as required by law or by order "
        "of a competent court. These obligations survive termination of this Agreement for five years. This Agreement "
        "is governed by the laws of England and Wales, whose courts have exclusive jurisdiction.",
        "The Fourth Amendment protects people against unreasonable searches and seizures, and a warrant must be "
        "supported by probable cause. In this case, the officers searched the defendant's phone without a warrant. "
        "The Supreme Court ruled that the search incident to arrest exception does not extend to digital data, and "
        "the evidence was suppressed. The conviction was vacated and the matter remanded.",
        "A contract requires an offer, acceptance, consideration and an intention to create legal relations. Where a "
        "party materially breaches a condition, the innocent party may treat the contract as discharged and sue for "
        "damages, subject to the duty to mitigate. Liquidated damages clauses are enforceable unless they impose a "
        "penalty out of proportion to the legitimate interest of the party relying on them.",
    ],
    "Finance": [
        "Revenue for the third quarter rose 12 percent year over year to 4.2 billion dollars, driven by subscription "
        "growth. Operating margin widened by 150 basis points, while free cash flow reached 610 million. The company "
        "repurchased shares worth 300 million and raised its full-year guidance. Analysts had expected earnings per "
        "share of 1.35, and the reported figure of 1.48 sent the stock up in after-hours trading.",
        "A bond's price moves inversely to interest rates. When the central bank raises its policy rate, newly issued "
        "bonds offer higher coupons, so existing bonds trade at a discount until their yield to maturity matches the "
        "market. Duration measures this sensitivity: a portfolio with a duration of seven years loses about seven "
        "percent of its value when yields rise by one percentage point.",
        "Inflation eased to 3.1 percent in March as energy prices fell, although core services remained sticky. Markets "
        "now price in two rate cuts this year, and the dollar weakened against the euro. Treasury yields declined "
        "across the curve, and credit spreads tightened. Economists warned that wage growth and a resilient labour "
        "market could delay any easing by the Federal Reserve.",
        "The balance sheet shows total assets of 18 million against liabilities of 11 million, leaving shareholders' "
        "equity of 7 million. Accounts receivable grew faster than sales, which suggests slower collections and "
        "pressure on working capital. Depreciation is charged on a straight-line basis, and the auditors noted a "
        "deferred tax liability arising from accelerated capital allowances.",
        "An index fund aims to match the market's return at low cost rather than to beat it. Over long periods, fees "
        "compound as powerfully as returns, so a one percent annual expense ratio can consume a quarter of the final "
        "balance. Diversification lowers company-specific risk, and rebalancing once a year between equities and bonds "
        "keeps the portfolio's allocation in line with the investor's tolerance for volatility.",
    ],
    "Fiction": [
        "The lamp had gone out hours ago, but Mara kept walking, one hand on the cold wall of the tunnel. Somewhere "
        "behind her, water dripped with the patience of a clock. She thought of her brother's laugh, of the way he had "
        "promised to wait at the gate. When the air finally changed and smelled of rain, she began to run, and she did "
        "not look back at the thing that whispered her name.",
        "\"You're late,\" said the old woman, without looking up from her knitting. Tom shook the snow from his coat and "
        "shut the door against the wind. The cat stretched on the hearth and gave him a long, unimpressed stare. \"The "
        "bridge is gone,\" he said quietly. She stopped, needles frozen in the firelight, and for the first time in "
        "forty years she looked afraid.",
        "On the morning the sea disappeared, the fishermen of Port Ellery stood on the harbour wall in their boots and "
        "said nothing at all. Boats lay tilted on the wet sand like sleeping dogs. Little Ines was the first to climb "
        "down, her red scarf snapping in the wind, and she walked out across the shining mud toward a light that "
        "nobody else would admit they could see.",
        "He had rehearsed the speech on the train, but when she opened the door with flour on her cheek, every word "
        "left him. Behind her, the kitchen radio played the song from that summer in Lisbon. \"Ten years,\" she said, "
        "and her voice was neither angry nor kind. He held out the letter, its corners soft from being carried, and "
        "waited for her to decide who they would be now.",
        "The dragon was smaller than the stories had claimed, and considerably more polite. It offered Sir Aldric a "
        "chipped teacup and asked whether he had come about the gold or the princess. \"Neither,\" the knight admitted, "
        "sinking onto a heap of coins with a sigh. \"My horse ran off, and I think I am lost.\" The dragon laughed so "
        "hard that smoke curled from its ears.",
    ],
}

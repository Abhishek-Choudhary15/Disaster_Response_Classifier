# EDA manual review notes (checklist items A15, B-manual, C4)

These items need a human. The notebook prints the random samples (fixed seeds); read them and record what you saw. Initial each section.

## HumAID - 15 random tweets per class (notebook section "B-manual")
| Class | Reviewer | Date | Observations (label noise? ambiguous? odd formatting?) |
|---|---|---|---|
| caution_and_advice | Abhishek Choudhary | 2026-10-31 | Almost all the tweets are correctly classified barring two: one is ambiguous and the other is just political criticism around disaster management. |
| displaced_people_and_evacuations | Abhishek Choudhary | 2026-10-31 | The label is applied to almost any tweet that mentions evacuation: 10 of 15 fit, 1 is clearly wrong (a joke about an Eagle evacuation) and 4 are ambiguous (thanks to hospital evacuation staff, a tax-relief notice for evacuees, criticism of the Houston mayor over evacuations, and an update with too little information). |
| infrastructure_and_utility_damage | Ankul Pal | 2026-10-31 | Almost all the tweets are correctly classified barring two: one is ambiguous (no specific damage described) and the other is really caution and advice (tips for filing an insurance claim). |
| injured_or_dead_people | Ankul Pal | 2026-10-31 | Almost all the tweets are correctly classified; only two are ambiguous - one mentions deaths not caused by a natural calamity and the other refers to deaths in a play about the Johnstown Flood. |
| missing_or_found_people | Himanshu Kasana | 2026-10-31 | Most tweets clearly report missing or unaccounted-for people. Some also mention death tolls or other disaster details, but the missing-person label remains relevant. |
| not_humanitarian | Himanshu Kasana | 2026-10-31 | Most tweets are unrelated personal posts or political opinions. One tweet about disaster assistance is incorrectly labelled, while the shelter headline is ambiguous. |
| other_relevant_information | Rohan Kumar | 2026-10-31 | 13 of 15 fit. The 2 misfits are a casual selfie post that belongs in not_humanitarian and a Shona/English government statement about food availability that is only loosely disaster information. This class is a catch-all that overlaps more specific classes (see the pair answer below). |
| requests_or_urgent_needs | Rohan Kumar | 2026-10-31 | 12 of 15 fit. 1 is clearly wrong (an unrelated crude joke about food) and 2 are ambiguous: a request to retweet with no stated need, and a remark that they need help that is really a political complaint about aid money. |
| rescue_volunteering_or_donation_effort | Sanjay | 2026-10-31 | 14 of 15 fit; fundraising events (a burger fundraiser, ticket sales for Red Cross relief) are correctly in this class. 1 is ambiguous because it is cut off mid-sentence, and the reviewer also noticed some tweets contain stray non-useful characters. |
| sympathy_and_support | Sanjay | 2026-10-31 | 14 of 15 fit. 1 is ambiguous: a personal remembrance of an activist that only mentions the California wildfires in passing. |

Ambiguous pair read side by side (notebook B7 cell): `not_humanitarian` vs `other_relevant_information` - how separable are they?

## Kaggle - random rows (A15) and hard negatives (C4)
- Reviewer / date: Rohan Kumar (random rows), Sanjay (hard negatives) 2026-10-31
- Do the hard negatives (non-disaster tweets with disaster words) look like genuine figurative uses?
- Any label that looks wrong?

## CrisisLexT26 - random rows (A15)
- Reviewer / date: Abhishek Choudhary 2026-10-31
- Non-English tweets noticed (see D7), garbled text, very short tweets:

## Answers to the template questions
- **not_humanitarian vs other_relevant_information - how separable are they?** The classes are generally separable. Unrelated personal posts and political opinions fit not_humanitarian, while useful disaster information fits other_relevant_information. Tweets mixing political commentary with relief information can be ambiguous, and some tweets fit more specific humanitarian classes. In the side-by-side sample, 5 of 10 not_humanitarian tweets were ambiguous, and 4 of 10 other_relevant_information tweets were wrong: three fit more specific classes (relief efforts, safety advice) and one was unrelated.
- **Kaggle: do the hard negatives look like genuine figurative uses?** Yes. After one verdict correction all 15 are genuine non-disaster uses, for example a baseball player who bombed one out, a cake that collapsed, a band called Buildings on Fire, a fire-department open house, and a prayer that God floods your life with blessings.
- **Kaggle: any label that looks wrong?** After correcting the verdicts, 1 clear error in the 15 random tweets: song lyrics labelled as a disaster. Three disaster-labelled tweets are debatable because they concern real-world events that are not a current disaster: a missing-man notice, a film release about the Chilean trapped miners, and an article on the Hiroshima bombings. Expect some label noise on the disaster side.
- **CrisisLexT26: non-English, garbled or very short tweets noticed?** Yes: 2 non-English (Spanish and Italian), 2 unrelated jokes (a Ragnarok game tweet and a typhoon-UFC pun) and 1 very short tweet ('ok... that was NOT cool!!! :( #earthquake'); the other 10 are readable English reports.

# Journey: Building the Encephalic Signal Distortion Classifier

This whole project was never just a "project" — it became a five‑month grind that went from confusion, to breakthroughs, to burnout, to hype, and finally to something I can actually ship and be proud of. I wanted to build something meaningful using EEG data, and I had no idea that meant wrestling with scientific datasets, machine learning pipelines, backend APIs, and front‑end design all at once.Yes specially no idea about EEG data , never thought it'd such a pain in the aah.

It started with the MODMA dataset — 128‑channel EEG data, stored in formats I had never seen before. Everything felt foreign in the beginning: channel names, montages, referencing systems, event markers… even loading a single file felt like a challenge. MNE‑Python was a whole new world of commands, warnings, and errors, and every step forward taught me something I didn’t know existed the previous day. But slowly, getting the montage right, filtering the signals, exporting clean `.fif` files, and batch‑processing the dataset became doable. That’s when the project shifted from "experimenting" to "building." Got backlash already for choosing to work with a dataset of this size and understood why very soon....

The whole pipeline of the model is fully Machine learning . No epoching no neural nets not that one . But...The first approach of classifying these signals were done through deep learning techniques . Through a span of 2-3 months , i built over 3 DL models hoping i could somehow reduce the MASSIVE overfitting .. but you know , when soomething is impossible , its never possible . yeah.

The machine learning part tested my patience in a completely different way. Riemannian geometry, CSP, covariance matrices — all of them sounded complicated, and honestly, they were. But after enough trial and error, the pipeline finally made sense. Tangent space projection + SVM became the first reliably performing classifier, and it felt good to have something that wasn’t random noise anymore. 

Then came the backend. FastAPI was new territory, but building an API that could accept `.raw` files, preprocess them, extract features, run inference, and send everything back as JSON was genuinely fun. At this point, the project felt like a real system, not just a notebook. Yep , was fascniated by all the abstract mechanisms humans built over the yeears(im a newbie).

And then… HuggingFace Spaces. This part deserves its own chapter. Every deployment attempt broke in a new way: missing dependencies, `405 Method Not Allowed`, model download issues, routing conflicts, UI not loading, file inputs triggering multiple times… the list goes on. But fixing these issues piece by piece taught me more about full‑stack deployment than anything else I’ve done before.

The frontend was its own journey. I designed a matte, glassy, neon UI from scratch using only HTML, CSS, and JavaScript. The upload box, result panel, glow effects . Connecting it smoothly to the backend, showing predictions, confidence scores, and detailed JSON output turned the project into something people could actually use.

Looking back, here are some of the biggest things this project made me learn:

* How to fully preprocess EEG data using MNE
* How classical ML (CSP, Riemannian geometry, SVM) works with biosignals
* How to design a clean inference pipeline
* How to build and deploy FastAPI services
* How to create a complete frontend from scratch
* How to debug deployment issues that have no obvious solution

It’s been five months of figuring things out one problem at a time, breaking things and fixing them, rewriting parts I thought were done, and constantly learning more than I expected. But now the entire system flows from raw EEG → cleaned signals → extracted features → classification → final UI output.

It’s not perfect, no , never , the model is so bad, but it’s real, it works, and I built the whole thing end‑to‑end. And that’s the part that matters the most. Tho it may not work as its supposed to , i learnt something useful , no , VERY useful.
Thankss :)

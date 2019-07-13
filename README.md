<p align="center">
  <img src="https://raw.githubusercontent.com/AndrewFerris/food_review/master/assets/logo.png" width="200" />
</p>

---

**FOOD REVIEW** is a simple classifier of sentiment for food reviews. It uses several out of the box classifiers to detect whether a review left had a positive or negative sentiment.

## Problem Description

The assignment covers designing and building the algorithm that will accept the text of any question and output the closest matching food review sentiment. The application should be a simple CLI tool which allows the user to input any question text. It should output the matching sentiment, with a relevance measure for each (an indicator of how well it matched). You are free to come up with a relevance measure, but provide a clear description of your selection.

Fundamentally the problem is, write a function that can be executed at the terminal with the following:
 - Inputs: a string of text, or a file with numerous strings
 - Outputs: Sentiment (Boolean, 1 for positive and 0 for not positive) and Confidence (Scale of 0 to 1 from least to most confident)

## Installation

## Methodology used by Food Review

The primary approach used by Food Review is to create a simple [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) model.
This converts a string into a corpus, which then allows the underlying text to be represented as an array of numerical data points.
From there it is possible to feed in any number of traditional classifiers into the model.
Below is a graphic which visually demonstrates the process.

<p align="center">
  <img src="https://raw.githubusercontent.com/AndrewFerris/food_review/master/assets/bag_of_words.jpg" width="600" />
</p>

Once this pre-processing was complete, I was able to use a wide range of classification models to help predict the sentiment of the review.
Some of the models that are used are below:

 - KNN
 - Random Forest
 - Naive Bayes
 - Regression
 - Tree & Forests
 - Support Vector Machine

 As well as stacking combinations of these together.

## Model Performance

Below are some high level performance results based on the data in this repository.

## Results

Below are the results of my best model on the test dataset. I haven't manually altered anything, but after visualling inspecting them I'm pretty happy with the results and I don't think it would need too much more work.

- Review: The string of text from the food review.
- Prediction: The model prediction output (1 for positive and 0 for not positive).
- Confidence of Positive (1): The likelihood the result was positive.
- Actual: Me reading the text an figuring out if it was actually positive or negative.

| Review | Predictions | Confidence of Positive (1) | Actual |
| ------ | ----------- | -------------------------- | ------ |
| There was a warm feeling with the service and I felt like their guest for a special treat. | 1 | 0.9267366849132375 | 1 |
| An extensive menu provides lots of options for breakfast. | 1 | 0.5588919228555813 | 1 |
| I always order from the vegetarian menu during dinner, which has a wide array of options to choose from. | 0 | 0.3131272840801199 | 1 |
| I have watched their prices inflate, portions get smaller and management attitudes grow rapidly! | 1 | 0.961884143623687 | 0 |
| Wonderful lil tapas and the ambience made me feel all warm and fuzzy inside. | 1 | 0.9801198760885833 | 1 |
| I got to enjoy the seafood salad, with a fabulous vinegrette. | 0 | 0.17557532373444878 | 1 |
| The wontons were thin, not thick and chewy, almost melt in your mouth. | 0 | 0.4103140892456718 | 0 |
| Level 5 spicy was perfect, where spice didn't over-whelm the soup. | 0 | 0.2662252080488948 | 1 |
| We were sat right on time and our server from the get go was FANTASTIC! |1 | 0.9356897688711006 | 1 |
| Main thing I didn't enjoy is that the crowd is of older crowd, around mid 30s and up. | 1 | 0.9601998742348041 | 0 |
| When I'm on this side of town, this will definitely be a spot I'll hit up again! | 1 | 0.571154997303588 | 1 |
| I had to wait over 30 minutes to get my drink and longer to get 2 arepas. | 0 | 0.14135547788002778 | 0 |
| This is a GREAT place to eat! | 0 | 0.16349827284579327 | 1 |
| The jalapeno bacon is soooo good. | 1 | 0.8259390883893538 | 1 |
| The service was poor and thats being nice. | 0 | 0.11080274116681552 | 0 |
| Food was good, service was good, Prices were good. | 0 | 0.15458623781847275 | 1 |
| The place was not clean and the food oh so stale! | 1 | 0.7179441757769099 | 0 |
| The chicken dishes are OK, the beef is like shoe leather. | 0 | 0.17316000128856301 | 0 |
| But the service was beyond bad. | 0 | 0.3241258728902579 | 0 |
| "I'm so happy to be here!!!""" | 0 | 0.43316575482913156 | 1 |
| Tasted like dirt. | 0 | 0.249202820392372 | 0 |
| One of the few places in Phoenix that I would definately go back to again . | 1 | 0.7871050976379358 | 1 |
| The block was amazing. | 1 | 0.7280306296811629 | 1 |
| It's close to my house, it's low-key, non-fancy, affordable prices, good food. | 1 | 0.854932766943981 | 1 |
| * Both the Hot & Sour & the Egg Flower Soups were absolutely 5 Stars! | 1 | 0.5714129795052706 | 1 |
| My sashimi was poor quality being soggy and tasteless. | 1 | 0.8942834370027358 | 0 |
| Great time - family dinner on a Sunday night. | 0 | 0.15105771822415207 | 1 |
| "the food is not tasty at all, not to say its ""real traditional Hunan style""." | 1 | 0.9969063606143154 | 0 |
| What did bother me, was the slow service. | 0 | 0.03585715169189131 | 0 |
| The flair bartenders are absolutely amazing! | 0 | 0.11475392471221255 | 1 |
| Their frozen margaritas are WAY too sugary for my taste. | 1 | 0.6965538526080328 | 0 |
| These were so good we ordered them twice. | 0 | 0.21957667887033902 | 1 |
| So in a nutshell: 1) The restaraunt smells like a combination of a dirty fish market and a sewer. | 1 | 0.7410718109117256 | 0 |
| My girlfriend's veal was very bad. | 0 | 0.18858400601142294 | 0 |
| Unfortunately, it was not good. | 1 | 0.6541122512106287 | 0 |
| I had a pretty satifying experience. | 1 | 0.6798737545290198 | 1 |
| Join the club and get awesome offers via email. | 1 | 0.9131016169218474 | 1 |
| Perfect for someone (me) who only likes beer ice cold, or in this case, even colder. | 0 | 0.15346520256824314 | 1 |
| Bland and flavorless is a good way of describing the barely tepid meat. | 1 | 0.9231947554112842 | 0 |
| The chains, which I'm no fan of, beat this place easily. | 0 | 0.2115190042509745 | 0 |
| The nachos are a MUST HAVE! | 1 | 0.7142355309484929 | 1 |

## Examples

To run this on a single review.

`python3 food_review_predict('This was a great meal, I really enjoyed eating here!')`

To run this on text file.

`python3 food_review_predict_file('~/path/filename.txt')`


## Other Possible Methodologies

 - [FastText](https://github.com/facebookresearch/fastText)
 - [TF-Hub](https://www.tensorflow.org/hub)
 - [GPT-2](https://openai.com/blog/better-language-models/)

**Why didn't I use any of the above?**
The main purpose of this package was to demonstrate my ability to develop a way to classify reviews. Most of the above are out of the box solutions that may or may not produce better results, but don't highlight my ability to do the work myself. If I was doing this as part of a professional project then this purpose would change.

**What other cases didn't you consider here?**
Many. Things such as sentiment posing, sarcasm, tone and higher order concepts weren't integrated.

**Why didn't you tune any hyperparameters?**
I could have applied a grid search approach for each of the techniques I used above.

**Why didn't you use other modelling techniques?**
Definitely could have used libraries such as Keras, PyTorch, etc. However the problem has some limitations in terms of dataset size to train on. What I could have done is downloaded the [Yelp food review dataset](https://www.yelp.com/dataset/documentation/main), which has over 6 million reviews and subsequent ratings, however this feels like it could be overkill for such a problem. It would have been an interesting approach to using transfer learning.

**Why didn't you do XYZ?**
1. This was for fun.
2. Given current work priorities I needed to complete it over the weekend.
3. I can always come back to it right? :)

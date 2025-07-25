
<div align="center">
<span style="color: crimson; font-weight: bold;">
<h1> Sentiment Analysis of 4 millions Amazon Reviews using Natural Language Processing </h1>
</span>
</div>


<div align="center">
<span style="color: crimson; font-weight: bold;">
<h3> Problem Statement </h3>
</span>
</div>
The project seeks to apply Natural Language Processing to Sentiment Analysis of a dataset containing 4 million Amazon reviews.


<div align="center">
<span style="color: crimson; font-weight: bold;">
<h3> Background </h3>
</span>
</div>

The vast amount of text data from reviews, discussions and X-posts, for example, have created a frenzy for companies to capitalize on this ocean of information for any actionable “intel.”  For example, restaurants are eager to know how their brand is faring in the public’s eyes.  Are foodies raving about their latest creation or are they lukewarm when it comes to the restaurant’s ambience?  Why do their patrons mostly skip the appetizers?  In finance, trading houses are keen on the latest X-post from Elon or any well-regarded financial pundits that can potentially send a stock to the moon or collapse it.  In retail like Amazon, it is critical to know what products are selling and not selling but more importantly, why?   What about the latest geo-political news? The latest trend posts from the Kardashians?  All these conversations, opinions and reviews have vast implications because of the speed which information travels and of the reach of social media.  Therefore, whether it is a restaurant or a fashion brand looking to improve their products or a trading house looking to improve their trading profit, sentiments play a key role in their success.

<div align="center">
<span style="color: crimson; font-weight: bold;">
<h3> Methodology </h3>
</span>
</div>

The project will focus on using Kaggle’s Amazon Reviews dataset to train, validate, test and evaluate Sentiment Analysis utilizing Natural Language Processing Models. Inference will be conducted on real-time reviews that users can input. The training, validating, testing, evaluating and inferencing will entail a supervised learning approach to a classification task.   The Kaggle dataset has “ground truth” that are correctly labeled.   The project will implement the underlying NLP models using self-trained and pre-trained deep models, mainly.  The project will not focus on shallow models such as Logistic Regression, Support Vector Machine or Naive Bayes, for example.   

<div align="center">
<span style="color: crimson; font-weight: bold;">
<h3> Data Selection </h3>
</span>
</div>

As mentioned previously, the project will be utilizing the 4 million Amazon Reviews dataset.   It is available from https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data   The dataset consists of 3.6M Amazon reviews in the training set and 400,000 in the testing set.


<div align="center">
<span style="color: crimson; font-weight: bold;">
<h3> Expected Outcomes </h3>
</span>
</div>

Upon project completion, various deep learning models will be able to infer real-time reviews which the users input into the demonstration application. 

Here are two examples. First is a positive Amazon review from the Amazon website:

[Original Amazon review](https://www.amazon.com/gp/customer-reviews/R2K8MBRGFPSZ3L)

Sentiment Analysis of the same Amazon review using the LSTM model:

![LSTM Amazon Review](images/demo-LSTM-pos-01.jpg)

Here is the second example. It is a negative Amazon review from the Amazon webiste:

[Original Amazon review](https://www.amazon.com/gp/customer-reviews/R1RLY0GI4V94MK)

Sentiment Analysis of the same Amazon review using the RNN model:

![Amazon Review](images/demo-RNN-neg-01.jpg)


# ml_examples

Small machine learning programs and examples built with numpy; sklearn used for benchmark scores only.

these are not built to be computationally efficient (obviously, if you look at them), but to get a better handle on how things work.

### Models:
##### Linear Regression
Rough analog to sklearn linear regression, using gradient descent.
##### Logistic Regression
Rough analog to sklearn logistic regression, using gradient descent.
##### Farklebot
learns to play the dice game Farkle

Scoring is hard coded; decisions about whether to re-roll or quit with current score are learned by playing a large number of games, altering strategy, and calculating the gradient of score with respect to strategic changes. Interesting because the gradient of decision making algorithm is not well behaved.


### Data visulaizations:
##### Linear regression
![linear results](https://github.com/notsambeck/ml_examples/blob/master/graphics/linear_regression_results.png)

##### Logistic regression
![logistic results](https://github.com/notsambeck/ml_examples/blob/master/graphics/logistic_regression_results.png)

##### Farklebot
![farkle_results1](https://github.com/notsambeck/ml_examples/blob/master/graphics/farkle_decision_boundary.png)
![farkle distribution](https://github.com/notsambeck/ml_examples/blob/master/graphics/farkle_optimal_score_distribution.png)

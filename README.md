Hi, the files that I would like to be graded are the two torch files and the pre-processing files. 

TLDR and Update: Please consider accepting my torch-trained model that I submitted late due to some
unfortunate events elaborated below. I would propose to be graded based on my torch-trained model,
however I do not want this to affect the other students, so I do not want to displace them on the
leaderboard, but I would like to be graded as if I was there.

Discussed with Prof. Guo and he will discuss it with the teaching team.
#################################################################################################
My original code that uses tensorflow/keras did not perform well. I spent a lot of time debugging 
the code and optimizing, thinking its just a matter of training. However, i later realized that 
the dictionary might be the issue, where I classify images incorrectly. I was training correctly
but training the wrong images.

Seeing my TF model not doing well, I had already created a script for training a model in pytorch.
I ran this model late, since the TA told me there should not be any difference in pytorch and
TF (This is correct, however the dataloader on pytorch seems to classify the things easier and
better personally, since I was able to get it right using pytorch.), I decided not to train my
data on pytorch, until the final submission slot.

For the final submission slot, 8-12PM on Thursday, I decided to run the pytorch code. Training went
well after a couple of hours. However for testing, I received a Python Fatal Error: Aborted. This
might be caused by me trying to install a different pytorch version to satisfy cuda (running on 
gpu). Installing and uninstalling torch did not work, and I tried multiple things as well, to no
success.

I decided to train the code with google colab instead, however it took some time to test. By the
time it finished training, it was unfortunately 12:01 AM. Despite so, I decided to submit the csv
file anyways at around 12:06 AM, barely missing the deadline. Please note if I had torch running 
on my computer, I would have been able to submit this file way earlier. Also, I had no prior 
submission for the 8-12 window, which means I would not be getting an extra submission. The csv 
file trained by torch received a 74% accuracy, while my TF code barely reached 9%.

It seemed nothing went well for me at that time, which was very unfortunate and it happens to us.
My request is for the teaching team to accept my torch trained results for grading. For the sake
of fairness to the other students, I do not want my score to be placed on the leaderboard and 
displacing a lot of students, potentially reducing their grades. However, I would like for my
grades to be evaulated according to my ranking if I was on the leaderboard.
################################################################################################

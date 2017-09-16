# gigers_lstm
This project aims to apply unsupervised and supervised learning methods on text data and utilise the resulting model for some useful applications:

* classification after an additional supervised training

* robust searching in the text

## Classifier

First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder (this is optional), then the resulting state is given to a classifier layer (with softmax), which calculates the estimated class.

## Autoencoder

First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder (this is optional), then the resulting state is given to an LSTM decoding RNN which tries to reconstruct the input sentence. After training the model is capable of transforming a given sentence into a dense code, which can be used for the mentioned applications.

## Results

### Classification

New music: <br />
Album Review: Jeff Tweedy – Together at Last <br />
Hudson Mohawke shares thumping new song “Passports” — listen <br />
Art-rock outfit Bent Knee share major label debut album Land Animal: Stream <br />
HAIM unveil studio version of “Little of Your Love” — listen <br />
Fischerspooner share first new song in eight years “Have Fun Tonight” co-written by Michael Stipe Boots — listen

New video: <br />
HAIM covers Shania Twain’s “Man! I Feel Like a Woman” — watch <br />
Fleet Foxes perform songs from new album Crack-Up on CBS This Morning — watch <br />
A kid befriends a robot soldier in ODESZA’s video for “Line of Sight” — watch <br />
CARACH ANGREN Posts Typically Creepy-As-Hell Video For Charles Francis Coghlan <br />
Príncipe's Puto Anderson releases video for 'Gritos Do Infinito'

Announcement: <br />
Jay Z announces new album 4:44 due out later this month <br />
Sälen announce free all-day party share acid-soaked pop number “So Rude” <br />
JAY-Z Announces New Album ‘4:44’ <br />
Bicep take their club love affair to the next level announce self-titled debut album on Ninja Tune <br />
Murlo: Club Coil EP

Tour / Festival / Concert: <br />
LCD Soundsystem announce new album American Dream plus lengthy tour <br />
A Perfect Circle announce North American arena tour for this fall <br />
New festival KALLIDA​ reveals lineup of visual artists for first edition <br />
Man Dies At EDC Las Vegas 2017 Of Apparent Heat-Related Issues <br />
Kaskade Ended Up Playing A Surprise Set at EDC Las Vegas 2017

Others: <br />
Supreme Court won’t hear infamous Prince vs Dancing Baby copyright case <br />
JAY-Z brings back the hyphen presses caps lock on his name <br />
Carrie Fisher had cocaine heroin and ecstasy in her system autopsy reveals <br />
The Slants win Supreme Court case over disparaging trademarks <br />
Chicago Rapper Jayaire Woods Gets Theatrical on “Big”

## Files

Note: If you want to use the pretrained checkpoints, you need to unzip ./checkpoint/autoencoder.zip and ./checkpoint/classifier.zip

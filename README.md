# gigers_lstm
This project aims to apply unsupervised and supervised learning methods on text data and utilise the resulting model for some useful applications:

* classification after an additional supervised training

* robust searching in the text

* to tell if two article titles are about the same subject (to be implemented)

## Classifier

First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder (this is optional), then the resulting state is given to a classifier layer (with softmax), which calculates the estimated class.

## Autoencoder

First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder (this is optional), then the resulting state is given to an LSTM decoding RNN which tries to reconstruct the input sentence. After training the model is capable of transforming a given sentence into a dense code, which can be used for the mentioned applications.

## Results

### Classification

The classifier is trained to decide if the news title is about new music, new video, announcement, tour / festival / concert, or about something else. A 90% accuracy can be achieved after training. Here are some results feeding in unlabelled data:

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
New festival KALLIDA reveals lineup of visual artists for first edition <br />
Man Dies At EDC Las Vegas 2017 Of Apparent Heat-Related Issues <br />
Kaskade Ended Up Playing A Surprise Set at EDC Las Vegas 2017

Others: <br />
Supreme Court won’t hear infamous Prince vs Dancing Baby copyright case <br />
JAY-Z brings back the hyphen presses caps lock on his name <br />
Carrie Fisher had cocaine heroin and ecstasy in her system autopsy reveals <br />
The Slants win Supreme Court case over disparaging trademarks <br />
Chicago Rapper Jayaire Woods Gets Theatrical on “Big”

### Autoencoder generates sentence from input sentence

These examples are the results after 1000 titles 'learned' by the autoencoder. The input sentence is fed in to the model, then the autoencoder encodes, then decodes it, and tries to reconstruct the input sentence.

Input sentence: <br />
Jay Z Announces New Album 4:44 Out Next Week <br />
Generated sentence: <br />
jay z announces new album before due next week this

Input sentence: <br />
Jay Z Announces New Album Next Week <br />
Generated sentence: <br />
jay z announces new album 97′s catches next week this

Input sentence: <br />
Jay Z Announces New Album <br />
Generated sentence: <br />
jay z announces new album before catches live at saint

Input sentence: <br />
Jay Z Announces <br />
Generated sentence: <br />
jay z announces new album before catches live at edc

Input sentence: <br />
Jay Z <br />
Generated sentence: <br />
jay z details new collaborative trials before before live at

### Search in the news

Search in the 1000 news titles and return the most relevant ones. The method is the following: <br />
All 1000 titles are encoded using the fully trained autoencoder. Then the search expression is encoded as well, the encoded search expression is compared to all the encoded titles, and the closest ones (cosine similarity) are selected as results.

Search expression: <br />
jay z announces album <br />
Search results (top 17): <br />
Jay Z Details New Album '4:44' <br />
Jay Z Details New Album '4:44' <br />
Jay Z announces new album 4:44 catches Magikarp at Barclays Center <br />
Jay Z announces new album 4:44 catches Magikarp at Barclays Center <br />
Jay Z Announces New Album 4:44 Out Next Week <br />
Jay Z announces new album 4:44 due out later this month <br />
Jay Z announces new album 4:44 due out later this month <br />
Jay Z’s new song “Adnis” soundtracks 4:44 trailer — watch <br />
Jay Z’s new song “Adnis” soundtracks 4:44 trailer — watch <br />
Titonton Duvante launches Residual Classic <br />
Gato Preto: Tempo (lemezkritika) <br />
Gato Preto: Tempo (lemezkritika) <br />
Deniro Farrar Drops ‘Mind Of A Gemini II’ Mixtape <br />
Afrojack Teases New Music Over Instagram Listen Here <br />
Tánci és röhögcse hollófeketében - Carach Angren-lemezpremier <br />
Nosaj Thing announces fourth album Parallels <br />
Palmbomen II releases Memories Of Cindy Pt. 2 from four-part series <br />

Search expression: <br />
arcade fire new single <br />
Search results (top 17): <br />
Arcade Fire share strobe-filled video for “Creature Comfort” — watch <br />
Arcade Fire share strobe-filled video for “Creature Comfort” — watch <br />
Arcade Fire share rousing new single “Creature Comfort” — listen <br />
Arcade Fire share rousing new single “Creature Comfort” — listen <br />
Dave East – “Only One King (EastMix)” <br />
Nick Höppner's favourite record sleeves <br />
Nick Höppner's favourite record sleeves <br />
Recapping Twin Peaks: The Return: Part 7 <br />
Recapping Twin Peaks: The Return: Part 7 <br />
Ilyen volt a Simple Plan a Budapest Parkban <br />
Ilyen volt a Simple Plan a Budapest Parkban <br />
Damon vízzel hint – Gorillaz-koncert a Várkert Bazárban (galéria) <br />
Damon vízzel hint – Gorillaz-koncert a Várkert Bazárban (galéria) <br />
Vadonatúj számokkal készül a Zagar a Kolorádóra <br />
Roger Waters’ new album blocked from release in Italy over alleged copyright infringement <br />
Roger Waters’ new album blocked from release in Italy over alleged copyright infringement <br />
Legújabb videójában felfedi titkait a The Weeknd <br />

Search expression: <br />
queens of the stone age new music <br />
Search results (top 17): <br />
Queens of the Stone Age unleash new single “The Way You Used to Do” — listen <br />
Queens of the Stone Age unleash new single “The Way You Used to Do” — listen <br />
Queens of the Stone Age announce 2017 North American tour with Royal Blood in support <br />
Queens of the Stone Age announce 2017 North American tour with Royal Blood in support <br />
Queens Of The Stone Age: The Way You Used To Do – az első teljes dal a Mark Ronsonnal készült albumról <br />
Queens Of The Stone Age: The Way You Used To Do – az első teljes dal a Mark Ronsonnal készült albumról <br />
Eagles of Death Metal is scoring the new Super Troopers movie <br />
Chance The Rapper stars in new Twitter Music campaign <br />
4 of the Best Looks From iHeartRadio Much Music Video Awards <br />
WREATH OF TONGUES live at Saint Vitus Bar Jun. 15th 2017 <br />
WREATH OF TONGUES live at Saint Vitus Bar Jun. 15th 2017 <br />
Stoned Love: The Stone Roses live in London <br />
Stoned Love: The Stone Roses live in London <br />
No I.D. to Produce JAY-Z's Entire '4:44' Album <br />
No I.D. to Produce JAY-Z's Entire '4:44' Album <br />
Paramore at the Royal Albert Hall in London <br />
Paramore at the Royal Albert Hall in London <br />

## Files

Note: If you want to use the pretrained model checkpoints, you need to unzip ./checkpoint/autoencoder.zip and ./checkpoint/classifier.zip

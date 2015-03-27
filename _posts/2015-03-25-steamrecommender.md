---
title: Steam Recommender
layout: post
---

## Steam Recommender

This post goes into some of the detail behind the
[steamrecommender.com](http://www.steamrecommender.com) site that generates
game recommendations based on your Steam play history. I have received
several questions about how I am able to generate the recommendations, so I
hope to give a fair account of the methods that I use.

If you have spent any time working on recommender systems, then you have
heard of the Netflix prize. In 2009, Netflix awarded the Belkor' Pragmatic
Chaos team with one-million dollars for a 10\% improvement in movie and TV
recommendations based on user ratings provided by Netflix. One outcome of
the Netflix prize is the raised awareness for recommender systems and
standardized ways of talking about recommendations.

Training data for the Netflix prize consisted of a collection of triples:
`(user, item, rating)` where the `user` referred to a unique user id
(anonymized of course) and the `item` is the movie or TV show being rated,
and `rating` being a quality score from one to five with five being best.
There are many open-source tools that work with this format or similar
formats, e.g. [GraphChi](https://github.com/GraphChi/graphchi-cpp) (with
work by [Danny
Bickson](http://bickson.blogspot.com/2012/12/collaborative-filtering-with-graphchi.html)),
[Prediction IO](http://prediction.io/),
[MLlib](https://spark.apache.org/mllib/), and others.  

In the end, I decided not to go with one of these existing services. This
project was for my personal enjoyment and so that I could learn more about
building web services, so I decided to build my own recommender system that
would plug in nicely with my service. In the next few sections I describe
different aspects of the system as a whole.  

### Data Ingestion

The recommendations generated from the steamrecommender.com site are
primarily based on an Amazon system that has been going strong for almost 20
years. You would recognize it as the panel beneath items in the Amazon store
with the title "Customers Who Bought This Item Also Bought." The idea is
that this panel would represent what your friends would recommend you
purchase, if your friends were the entire user base of Amazon.  

In order to generate this type of personalized recommendation for video
games, I searched for datasets where I could gather all of the games owned
by a person and how well they liked each game. The Steam store by Valve came
the closest, since they provide a web API for developers to build
integrations. In a piece by Kyle Orland published on [Ars
Technica](http://arstechnica.com/gaming/2014/04/introducing-steam-gauge-ars-reveals-steams-most-popular-games/),
he describes a technique for querying this API for random Steam users to
build up a dataset. With this technique we are able to get all of the games
owned by a Steam user along with the number of minutes spent playing the
game in the last two weeks as well as total time spent playing the game.
Using similar techniques I am able to gather a sample of roughly 50k players
every day.  

### Model Building

The data ingestion system runs uninterrupted daily and gathers more and more
of these training samples. After a month of gathering publicly accessible
Steam accounts I have roughly 1.5 million profiles to construct
recommendations from. It is important to note that Steam users still have
control over their privacy. At any time they can set their profile to
private and I would not have any access to the games they own nor would I
have any access to the amount of time they have spent playing those games.  

The recommendation engine powering
[steamrecommender.com](http://www.steamrecommender.com) is based on the
original paper: [Amazon.com Recommendations: Item-to-Item Collaborative
Filtering](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1167344).
The idea is to construct a matrix wherein the rows are the Steam users and
each column is a different steam game. The challenge is deciding what the
rating should be for a user and a game. We briefly explore the different
options and the intuition for what each entails below.  

1. **ownership**: In the ownership rating system, each cell in the matrix
has a one in it if the user owns the game and a zero if the user does not.
Under this assumption, popular games are often recommended more frequently
because they happen to be owned by a larger number of people. Relying on
ownership for a rating is limited. Just looking through my own Steam
library, I realized that I had a rather large backlog of games that I owned
by had not yet played. If you were to personally ask me about those games, I
would not be able to recommend them because I had not played them. This
insight led to my second option for a rating.  

2. **played**: In the played rating system, each cell in the matrix has a
one in it if the user has played the game and a zero otherwise. The user
must have spent at least 30 minutes in the game before we consider the game
played. If you quit a game prior to spending 30 minutes in the game, you
either hate the game or are saving it to come back later. In either case, I
see that as the user not wanting to recommend that game. Recommendations
from the played system are qualitatively better than those from the
ownership system since we are better able to filter out games that impulse
purchases from Steam sales and have a better signal to remove disliked
games.  

3. **inferred**: For the inferred ratings, I go one step further and try to
compute a 5-star rating for each game from your playtime. My 5-star rating
system will assume you would give a 5-star rating to a game if you played it
to completion or longer. I make some assumptions (that may or may not be
true) in order to create inferred ratings. First, I assume that majority of
players will not finish a game. They may become bored, or may dislike the
game, but either way they will not play through to completion. Second, I
assume that there are some die-hard players who will continue to replay a
game over and over simply because they love the game. Based on these two
assumptions, I decided that the 75th percentile of the total time spent
playing distribution (across all steam users who played the game) would
represent the expected playtime of a game that you loved playing and would
happily recommend.  

For every player, the inferred rating is computed by dividing the number of
minutes spent playing teh game with the 75th percentile for that game and
then multiplying by five (for five stars), keeping the maximum star rating
at 5. For example, if you played a game for 120 minutes and the 75th
percentile is 280 minutes, then the 5-star rating for this game would be
`(120 / max(280, 120) * 5) = 2`.

The **inferred** rating system is the one that is currently deployed on the
website as it generated the best qualitative results. Once you have a method
for computing the ratings, building a model for generating recommendations
is straightforward. I will try and provide the intuition behind the
collaborative filter that I build and leave the details to the original
[paper](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1167344).
For a more gentle introduction to recommendation techniques, I recommend
[Programming Collective
Intelligence](http://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325).

To generate recommendations for a single user, we first gather the games
that he/she has played by making a call to the Steam web API. The playtimes
associated with each game are converted into inferred ratings and every game
that has not been played is dropped. At this point we should have a
collection of games that our Steam user has played enough to generate a
rating. What we would like is a principled way to find games similar to
these that other users have also found favorable.  

That is where the power of the collective comes in handy. We iterate through
all of the players and look at the pairs of games that they have played. If
a player has two games that they have played and given high ratings to, then
odds are another player who has played one of those games will enjoy the
other one. If you do that over 1.5 million players, this signal becomes
pretty clear. We convert this intuition into something more formal by
defining a distance metric between two games as the *cosine* similarity
between the rating vectors. An example will hopefully help clear up any
confusion.

| | [Skyrim](http://store.steampowered.com/app/72850/) | [Bioshock Infinite](http://store.steampowered.com/app/8870/) | [Left 4 Dead 2](http://store.steampowered.com/app/550/) | [Civilization V](http://store.steampowered.com/app/8930/) |
------|:---:|:---:|:---:|:---:|
Jim   | 5   | 5   |     | 3   |
Sally |     | 5   | 5   |     |
Gina  | 2   | 3   | 3   | 3   |
Bob   | 1   | 1   | 1   | 1   |

In this example, we can see that Jim either does not own or has not played
Left 4 Dead 2 (L4D2) and the same goes for Sally with Skyrim and
Civilization 5. Both Gina and Bob have played all of the games, but it seems
that Bob cannot find anything that he really likes and tends to play games
for short periods of time. Our recommendation system relies on recommending
similar games to the ones that you have played before, so we will compute
the [cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity)
between all pairs of games. some notion of similarity. For example, we want
to compute the similarity between L4D2 and Bioshock. The rating vectors for
these two games are: `[5, 3, 1]` and `[5, 3, 1]`. Jim has not played both
games, so we do not include his rating in this computation. This example is
not particularly interesting since the vectors are identical, the similarity
between them is one. Of course with more examples, this would not be the
case.

The result is that we have a matrix of similarities between games. A single
column represents a single game and the similarity between it and all of the
other games. If someone had only played Sid Meier's Civilization V and
he/she was looking for new recommendations, then we could order this column
by the most similar games to Civilization 5 and generate decent
recommendations (Note: these recommendations actually came from http://steamrecommender.com on 03/26/2015).  

<center>
<figure>
<img src="http://cdn.akamai.steamstatic.com/steam/apps/8930/header.jpg?t=1414605253" alt="" width="250px"><br><a href="http://store.steampowered.com/app/8930/">Sid Meier's Civilization V</a>
</figure>

<hr/>

<img src="http://cdn.akamai.steamstatic.com/steam/apps/65980/header.jpg?t=1424967897" alt="" width="250px"><br><a href="http://store.steampowered.com/app/65980">Sid Meier's Civilization: Beyond Earth</a>
<br><br>
<img src="http://cdn.akamai.steamstatic.com/steam/apps/289130/header.jpg?t=1427213505" alt="" width="250px"><br><a href="http://store.steampowered.com/app/289130">Endless Legend</a>
<br><br>
<img src="http://cdn.akamai.steamstatic.com/steam/apps/24780/header.jpg?t=1405370329" alt="" width="250px"><br><a href="http://store.steampowered.com/app/24780">Sim City 4: Deluxe Edition</a>
<br><br>
<img src="http://cdn.akamai.steamstatic.com/steam/apps/10500/header.jpg?t=1420554760" alt="" width="250px"><br><a href="http://store.steampowered.com/app/10500">Empire: Total War</a>
</center>

If someone has played more than one game, we weight each column by the
rating that the user gave to that particular game, sum across of the games
the user has played and sort to find the most similar games to this
particular user.

### Conclusions

This post has taken longer than I had originally planned an glosses over some
details while getting into the gory details for others.  Hopefully this post
mixed with some external reading will make everything clear.  

If you have built recommendations before, or happen to be a researcher in
the field, you may wonder why I am using technology from the early 2000s
instead of using methods like matrix factorization that helped win the
Netflix prize. The primary reason is that I needed to be able to generate
recommendations for users who were not in my sample set. I would have no
guarantees that I have a factorization for a new user as they may have not
been part of my initial training set. Of course, I could always add them to
the set and redo the factorization, but that takes time and would result in
a poor user experience.  

The original idea for http://steamrecommender.com was to build a website
that showcased recommendations for a product area that I thought was
woefully neglected. At the time, Valve had not launched their updated
recommendation system and I often wondered what I should play next. At the
time, Steam sales were how I would decide what to play next because I would
often purchase whatever was on sale. Given my background in recommender
systems I thought that there had to be a better way. Now the site has become
a fun toy for me to think about different recommender challenges and
sometimes discover new games to play. 

Now that this post is finished, I will begin work on the matrix
factorization blog post where I discuss computing game embeddings and using
them to improve discoverability, search, and recommendations.  

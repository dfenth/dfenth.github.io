---
layout: post
title: "Bluesky Search"
date: 2026-05-23 12:00:00 +0000
categories: projects ml
usemathjax: false

---

The idea is simple. Given a user with their general "Following" feed (the one that collates the posts from everyone you follow), search the feed for information that matches closely with the user input.

There are a few ways of doing this. In the pre-AI era, this could be done using keywords, with high-ranking results containing more of the searched keywords than lower-ranking ones, but this isn't really how things are done anymore. More modern approaches use LLMs, leveraging their ability to understand context, so if we search for "fast food chips", we'll have high-ranking food results rather than computer chips.

With LLMs in mind, my approach to search is:
- Take each post from my "Following" feed
- Use an LLM to embed the post in an embedding space
- Search for something, being vague or specific
- Return the posts that are close to the search text in the concept space

This is the code:
```python
from atproto import Client

import os

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# Check environment variables for login exist
try:
	username = os.environ['BSKY_USERNAME']
	password = os.environ['BSKY_PASSWORD']
except Exception:
	raise RuntimeError(
		"Credentials not set in appropriate environment variables!"
	)

# Login
client = Client()
profile = client.login(username, password)
print("Logged in as {}".format(profile.display_name))

# Get the target timeline
target_feed = client.app.bsky.feed.get_timeline().feed 

# We should get the latest 50 posts
print("Feed length: {}".format(len(target_feed)))

posts = []
  
for p in target_feed:
	uri = p.post.uri.split('/')[-1]
	# Get the post text
	text = p.post.record.text
	
	try:
		# Add descriptions if the post has an external link
		text += " - " + p.post.embed.external.description
	except:
		pass
	
	posts.append(text)

# Embed the posts using a sentence transformer  
embedding_model = SentenceTransformer(
	'sentence-transformers/all-mpnet-base-v2'
)

# Create the SentenceTransformerEmbeddings wrapper
embedding_function_wrapper = SentenceTransformerEmbeddings(
	client=embedding_model
)

# Initialise Chroma using the from_texts class method
# Chroma embeds the posts and gives us some nice functions for top-k retrieval
vector_db = Chroma.from_texts(
	collection_name="post_embeddings",
	texts=posts,
	embedding=embedding_function_wrapper,
)

# Search!
while True:
	query = input(">>")
	res = vector_db.similarity_search_with_score(query, k=10)
	
	# Print the top-k results
	for r, s in res:
		print(r.page_content)
		print("Score: {:.2f}".format(s))
		print("="*20)
```

For this, we're using the [AT Protocol SDK](https://atproto.blue/en/latest/) to access Bluesky, logging in using a username and password defined in environment variables (`export BSKY_USERNAME="some_username"` and `export BSKY_PASSWORD="some_password"` when using Ubuntu):
```python
# Check environment variables for login exist
try:
	username = os.environ['BSKY_USERNAME']
	password = os.environ['BSKY_PASSWORD']
except Exception:
	raise RuntimeError(
		"Credentials not set in appropriate environment variables!"
	)

# Login
client = Client()
profile = client.login(username, password)
print("Logged in as {}".format(profile.display_name))
```
Followed by the processing of posts in the feed, which is fairly self-explanatory once you get the right AT protocol function calls, which are not always easy.

The other interesting part is the core of the search function, which uses embeddings of the posts and the query. We define the embedding model:
```python
# Embed the posts using a sentence transformer  
embedding_model = SentenceTransformer(
	'sentence-transformers/all-mpnet-base-v2'
)

# Create the SentenceTransformerEmbeddings wrapper
embedding_function_wrapper = SentenceTransformerEmbeddings(
	client=embedding_model
)
```
This is a specific model built to embed sentences ([all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)), which handles a lot of the annoying parts you'd need to implement if you were using a standard LLM, such as chunking the post (so context can be retained across all of the text), and the output being a vector embedding (rather than the generation of new text).

The embeddings are then stored using Chroma:
```python
vector_db = Chroma.from_texts(
	collection_name="post_embeddings",
	texts=posts,
	embedding=embedding_function_wrapper,
)
```
[Chroma](https://reference.langchain.com/python/langchain-chroma) is a vector store, designed for exactly this kind of work. The LLM generates the vector representation of the post, we store it using chroma, and we can leverage built-in functions to determine similarities between the stored posts and query text:
```python
res = vector_db.similarity_search_with_score(query, k=10)
```
The [documentation](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma/similarity_search_with_score) is pretty bad and doesn't really define the similarity metric or the score. You'd usually use something like cosine similarity when comparing vectors, but high-scoring samples seem to return numbers around 1.5-1.6, which don't look very cosine.

Anyway, it seems to work okay! The posts that surface from a query are relevant, and on multiple occasions, I've been able to find a post I'd been thinking about using a somewhat general query (i.e. not using unusual words that would only appear in that post).

An obvious restriction of this code, as it stands, is that it only takes the most recent 50 posts from your feed. This could quite easily be scaled to search through your entire feed history, but it could take some time to process (embedding each post individually and storing them in Chroma), and while you're able to access feed histories in Bluesky, I'm not sure how far back they go! But the core methodology of this experiment (LLM post encoding followed by search using Chroma) is built to scale. If we added code to save the vector database to disk and added some feed handling to update the vector store with new posts, we'd have a pretty cool Bluesky search function!
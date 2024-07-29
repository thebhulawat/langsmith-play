from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
response = remote_chain.invoke({"language": "hindi", "text": "I believe in myself to be founder of a great company"})

print(response)
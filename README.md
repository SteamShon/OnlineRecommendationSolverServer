This is server implementation for interactive label propagation algorithm on http://1.234.62.79:5000.
currently above demo site implemented following two algorithm for song recommendations.


1. ALS matrix factorization.
	1.1 Matrix M(item x latent features) file ([item_id\tvector of latent features] format).
	
2. Label propagation.
	2.1 User x Item rating file([user_id, item_id, rate] format).

above two algorithm load each files into memory on server startup, and for user`s new rating vector, it calculate algorithms and return recommendations.

Jetty is used for this simple solver server.


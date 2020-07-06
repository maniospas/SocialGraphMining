# Social Graph Mining Module

## Introduction
This module aims to provide dynamic machine learning capabilities to HELIOS users.
In particular, each user (or, more precisely, their HELIOS device) carries a different instance of this module 
and needs to account for its capabilities to facilitate recommendation tasks. Recommendations are performed on a per-context
basis and can facilitate various objectives.

## Installation
This module depends on `eu.h2020.helios_social.core.contextualegonetwork`.

### Jar File installation
This project can be downloaded as a [jar file](../jar/h.extension-SocialGraphMining 1.0.0.jar), which can be added on a
Java project's dependencies. This requires also downloading the respective [ContextualEgoNetwork JAR]().

### Gradle Installation
##### First step
Add the JitPack repository to your build file. In particular, add it in your root build.gradle at the end of repositories:
```
allprojects {
	repositories {
		...
		maven { url 'https://jitpack.io' }
	}
}
```
##### Second step
Add the dependency:
```
dependencies {
	implementation 'com.github.User:Repo:Tag'
}
```

### Maven Installation
##### First step
Add the JitPack repository to your build pom file:

```xml
<repositories>
	<repository>
	    <id>jitpack.io</id>
	    <url>https://jitpack.io</url>
	</repository>
</repositories>
```

##### Second step
Add the dependency:

```xml
<dependency>
    <groupId>com.github.DistributedSystemsSocialNetworkAnalysis</groupId>
    <artifactId>Contextual-Ego-Network</artifactId>
    <version>Tag</version>
</dependency>
```

## API Usage
Here we detail how to develop applications using this module's graph API to provide social graph recommendation capabilities.

### Instantiating the graph miner
Setting up graph mining capabilities start from a contextual ego network instance. If such an instance is not available,
it can be created when the application starts using the following code:
```java
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
    
String userid = "ego"; // can use any kind of user id for the ego node
String internalStoragePath = app.getApplicationContext().getDir("egonetwork", MODE_PRIVATE); // an internal storage path in Android devices
ContextualEgoNetwork contextualEgoNetwork = ContextualEgoNetwork.createOrLoad(internalStoragePath, userid, null);
```
The contextual ego network library is used by this module to attach information on the perceived
social graph structure and can be saved using the command `contextualEgoNetwork.save()`. For a more
detailed description see the documentation of the library. Only a **singleton** ContextualEgoNetwork instance should be created 
in an application and it should be shared with any other modules that potentially depend on it.

Given the contextual ego network instance, we can then set up a switchable miner to be able to alternate between different mining algorithms in runtime (e.g. to change the criteria interaction recommendations are sorted by). Here we will show how to alternate between two types of graph miners: one that re-recommends recent interactions (RepeatAndReplyMiner) and one that uses GNNs to
recommend interactions (GNNMiner). Initializing a switchable miner and adding these two types of miners on it can be done with
the following code:

```java
import eu.h2020.helios_social.modules.socialgraphmining.SwitchableMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNMiner;

SwitchableMiner miner = new SwitchableMiner(contextualEgoNetwork);
miner.createMiner("repeat", RepeatAndReplyMiner.class);
miner.createMiner("gnn", GNNMiner.class).setDeniability(0.1, 0.1); //also apply 10% differential privacy and plausible deniability
miner.setActiveMiner("gnn");
```
Alternating between the created miner can be done by referencing their given names through the commands
`miner.setActiveMiner("repeat");` and `miner.setActiveMiner("gnn");` respectively. This alteration changes only which of the
two miners are used for recommendation (see below) but simultaneously trains all of them after the switchable miner has been
notified of a new interactionr. In the above example we set the GNN miner as the type of miner the application starts recommending
with.

It must be noted that, after instantiating a graph miner, such as the switchable miner, it needs to be constantly notified about
user interactions and somes exchange parameters with other devices (see below).

### Recommending interactions in the current context
Before explaining how to train the miners, it must be pointed out that training and predictions change as user switch contexts.
In HELIOS, multiple contexts (e.g. home, work) may be defined by each users. In case where context switching is a not a
necessity in an application, a generic purpose context can be created and set up as the current context (i.e. so as to be easily
retrievable). We remind that doing so using the contextual ego network management library can be achieved through the following code:

```java
contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
```

Then, using a social graph miner to obtain social recommendations for nodes of the contextual ego network in the current context can be achieved through the following code:
```java
import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Node;

Context context = contextualEgoNetwork.getCurrentContext(); // can also obtain other contexts to recommend for
HashMap<Node, Double> interactWithNodeRecommendation = miner.recommendInteractions(context);
```
The recommendations are (Node, weight) entries for all nodes in the current context, where weight values lie in the range [0,1]
with higher ones indicating stronger recommendation for interacting with the respective node. 

### Communication scheme
A requirement for using the social graph mining algorithms is that they need to exchange information when social interactions occur. **Not doing so will considerably worsen the quality of some mining algorithms**, especially those based on GNNs. Our design requires
only little communication (i.e. three information exchanges) only when the interactions occur and of few parameters (e.g. at worst, expect a 100 double numbers converted to strings).

Our algorithms assume that, when the user of a device *A* initiates a social interaction towards the user of a device *B* the following communication steps are followed:

##### First step (SEND)
*A* creates an interaction in its instance of the contextual ego network retrieves a set of parameters (parametersOfA) from its miner given that interaction:
```java
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;

Interaction interaction = A.contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(nameOfB, null))
				.addDetectedInteraction(null);
String parametersOfA = A.miner.getModelParameters(interaction);
```
Then *A* sends to *B* its parameters (e.g. by attaching them on the sent message or immediately after the sent message).

##### Second step (RECEIVE)
*B* receives the parameters of *A* (parametersOfA), creates a new interaction on its instance of the contextual ego network, notifies its miner about the receive and creates a new set of parameters (parametersOfB):
```java
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;

Interaction interaction = B.contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(nameOfA, null))
				.addDetectedInteraction(null);
B.miner.newInteraction(interaction, parametersOfA, InteractionType.RECEIVE);
String parametersOfB = miner.getModelParameters(interaction);
```
Then *B* sends back to *A* its parameters (e.g. by attaching to a receive acknowledgement message).

##### Third step (RECEIVE_ACK)
*A* receives the parameters of *B* (parametersOfB), retrieves the interaction these refer to and notifies its miner about the update:

```java
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;

ArrayList<Interaction> edgeInteractions = A.contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(A.contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(nameOfB, null))
				.getInteractions();
Interaction interaction = interactions.get(interactions.size()-1);
miner.newInteraction(parametersOfB, InteractionType.RECEIVE_REPLY);
```

## Project Structure
This project contains the following components:

src - The source code files.

doc - Additional documentation files.
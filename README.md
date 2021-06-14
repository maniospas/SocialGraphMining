# Social Graph Mining Module

## Introduction
This module aims to provide dynamic machine learning capabilities to HELIOS users.
In particular, each user (or, more precisely, their HELIOS device) carries a different instance of this module 
and needs to account for its capabilities to facilitate recommendation tasks. Recommendations are performed on a per-context
basis and can facilitate various objectives.

## Installation
[![](https://jitpack.io/v/helios-h2020/h.extension-SocialGraphMining.svg)](https://jitpack.io/#helios-h2020/h.extension-SocialGraphMining)
This module depends on [eu.h2020.helios_social.core.contextualegonetwork](https://githuBob.com/helios-h2020/h.core-SocialEgoNetwork).

### Jar File installation
This project can be downloaded as a [jar file](https://githuBob.com/helios-h2020/h.extension-SocialGraphMining/blob/master/jar/h.extension-SocialGraphMining1.0.3.jar), which can be added on a
Java project's dependencies. This requires also adding the Jar of the ContextualEgoNetwork library.

### Gradle Installation
First, add the JitPack repository to your build file. In particular, add it in your root build.gradle at the end of repositories:

```
allprojects {
	repositories {
		...
		maven { url 'https://jitpack.io' }
	}
}
```

Then add the dependency:

```
dependencies {
        implementation 'com.githuBob.helios-h2020:h.extension-SocialGraphMining:1.0.4'
}
```

### Maven Installation
First add the JitPack repository to your build pom file:

```xml
<repositories>
	<repository>
	    <id>jitpack.io</id>
	    <url>https://jitpack.io</url>
	</repository>
</repositories>
```

Then add the dependency:

```xml
<dependency>
    <groupId>com.githuBob.helios-h2020</groupId>
    <artifactId>h.extension-SocialGraphMining</artifactId>
    <version>1.0.4</version>
</dependency>
```

## API Usage
Here we detail how to develop applications using this module's graph API to provide social graph recommendation capabilities.

### Instantiating the graph miner
Graph mining capabilities start from a contextual ego network instance. If such an instance is not available,
it can be created when the application starts with the following code:

```java
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
    
String userid = "ego"; // can use any kind of user id for the ego node
String internalStoragePath = app.getApplicationContext().getDir("egonetwork", MODE_PRIVATE); // an internal storage path in Android devices
ContextualEgoNetwork contextualEgoNetwork = ContextualEgoNetwork.createOrLoad(internalStoragePath, userid, null);
```

The contextual ego network library is used by this module to attach information on the perceived
social graph structure and can be saved with the command `contextualEgoNetwork.save()`. For a more
detailed description, see the respective documentation Only a **singleton** ContextualEgoNetwork instance should be created 
in an application and it should be shared with any other modules that potentially depend on it.

Given a contextual ego network instance, it is recommended set up a switchable miner, which holds any number of mining algorithms. Switchable miners can alternate between held miners at runtime, for example to change the criteria interaction recommendations are sorted by. Typically, only a **singleton** switchable miner will be defined per android application. Here we will show how to alternate between two types of graph miners: one that re-recommends recent interactions (RepeatAndReplyMiner) and one that uses GNNs to
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
Alternating between the above created miners can be done through their names through the commands
`miner.setActiveMiner("repeat");` and `miner.setActiveMiner("gnn");` respectively. This alternation changes only which of the
two miners is used for recommendation (see below) but simultaneously trains all of them whenever the switchable miner is
notified of new interactions. In the above example we set the GNN miner as the type of miner the application starts recommending
with.

It must be noted that, after instantiating a graph miner, such as the switchable miner, it needs to be constantly notified about
user interactions and some exchange parameters with other devices (see below).

### Recommending interactions in the current context
Before explaining how to train the miners, it must be pointed out that predictions change as users switch contexts.
In HELIOS, multiple contexts (e.g. home, work) may be defined by each user. In case where context switching is a not a
necessity in an application, a generic-purpose context can be created and set up as the current one.
Doing this with the contextual ego network management library can be achieved through the following code:

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
A requirement for using the social graph mining algorithms is that they need to exchange information when social interactions occur. **Not doing so will considerably worsen the quality of some mining algorithms**, especially those based on graph diffusion or GNNs. Our design 
requires little communication (i.e. three information exchanges), only when the interactions occur and of few parameters (e.g. at worst, expect 100 double numbers converted to strings).

:bulb: Switchable miners automatically exchange parameters for all registered miners. As such, the following communication protocol
needs to be implemented **only** for the switchable miner holding all application miners.

Our algorithms assume that, when the user of a device *A* initiates a social interaction towards the user of a device *B* the following communication steps are followed:

##### First step (SEND)
*A* creates an interaction in its instance of the contextual ego network retrieves a set of parameters (parametersOfAlice) from its miner given that interaction:
```java
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;

Interaction interaction = Alice.contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(nameOfB, null))
				.addDetectedInteraction(null);
String parametersOfAlice = Alice.miner.getModelParameters(interaction);
```
Then *A* sends to *B* its parameters (e.g. by attaching them on the sent message or immediately after the sent message).

##### Second step (RECEIVE)
*B* receives the parameters of *A* (parametersOfAlice), creates a new interaction on its instance of the contextual ego network, notifies its miner about the receive and creates a new set of parameters (parametersOfBob):
```java
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;

Interaction interaction = Bob.contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(nameOfA, null))
				.addDetectedInteraction(null);
Bob.miner.newInteraction(interaction, parametersOfAlice, InteractionType.RECEIVE);
String parametersOfBob = miner.getModelParameters(interaction);
```
Then *B* sends back to *A* its parameters (e.g. by attaching to a receive acknowledgement message).

##### Third step (RECEIVE_ACK)
*Alice* receives the parameters of *Bob* (parametersOfBob), retrieves the interaction these refer to and notifies its miner about the update:

```java
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;

ArrayList<Interaction> edgeInteractions = Alice.contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(Alice.contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(nameOfB, null))
				.getInteractions();
Interaction interaction = interactions.get(interactions.size()-1);
miner.newInteraction(parametersOfBob, InteractionType.RECEIVE_REPLY);
```

### Diffusing predictions through the decentralized social graph
The mining module can be used to augment the predictive capabilities of other modules through graph diffusion.
This is achieved through the `PPRMiner`, which implements a decentralized version of the random walk with restart scheme
to aggregate predictions across ego network alters with the ego's device predictions. To use this miner, we consider a
scheme where each device makes its own prediction about its users (e.g. a classification of their interests), where
predictions are **not** necessarily made for all devices, for instance due to lack of data features. At worst, some 
device users could have manually classified themselves.

##### User personalization vector
In this setting, we consider that each device encodes its user's information into a *personalization vector*, which is an
one-hot encoding of their classification label. For example, given that the device's user is assigned to the class
 *cl=0,1,...,n-1* out of *n* potential classes, where classes are consecutive integer numbers. That is, the first class
is assigned label 0, the second label 1, the third label 2 and so on.
These labels can be converted to an one-hot encoding, which can be obtained per
`personalization = new DenseTensor(n).put(cl,1);` where `new DenseTensor(n)` creates a *n*-dimensional vector data structure
to manipulate vectors provided by the JGNN library and `put(cl, 1)` puts value *1* at position *cl*. 
For users with unknown classification labels, the personalization vector comprises only zeros and can be constructed per 
 `personalization = new DenseTensor(n);`, that is without using the put method to assign any values.
 
Although the above formulation refers to non-overlapping classes, in principle the personalization vector can hold any
non-negative values at its elements, for example obtained as probabilities of .
 
##### PPRMiner to disseminate information
Then, in each device, the miner needs to be constructed given a unique name (that differentiates between multiple 
usages by different modules) the contextual ego network instance to attach information to and the personalization vector
for the device's user per `pprMiner = new PPRMiner(name, contextualEgoNetwork, personalization)`.
The personalization can be updated later on through the `pprMiner.updatePersonalization(personalization);` method.
To further prevent diffusion to change non-zero (i.e. known) personalizations, the PPRMiner can be made to forcefully
keep known predictions through `miner.setPersonalizationAsGroundTruth(true);`

As a final note, PPRMiners extend the social graph mining class and hence need to exchange parameters with their
neighbors given the previous communication scheme. For example usage of the PPRMiner, please refer to the
simulation code at *eu.h2020.helios_social.modules.socialgraphmining.experiments.DiffusionSimulation.java*.
For applications that already implement a switchable miner and have implemented a communication scheme for the latter,
a new PPRMiner instance can be explicitly registered with the following code:

```Java

SwitchableMiner miner = ... ; // obtain the switchable miner
PPRMiner pprMiner = ...; // the instantiated PPRMiner
miner.registerMiner(pprMiner.getName(), pprMiner);
```

In this 

While mining takes place, current estimation of smoothed class scores can be retrieved with `miner.getSmoothedPersonalization();`.
This returns a vector with equal size to the personalization vector.


## Project Structure
This project contains the following components:

src - The source code files.

doc - Additional documentation files.

jar - Jar file installation.
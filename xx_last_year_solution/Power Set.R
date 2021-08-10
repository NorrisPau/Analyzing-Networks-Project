#--------------------------------------------------------------------------------------------------#
#---- POWER SET ALGORITHM -------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
set <- c("a","b","c")

pset <- list()
pset[[1]] <- set[1]

for (i in 2:length(set)){
  
  psetCopy <- pset
  
    for (j in 1:length(psetCopy)){
      psetCopy[[j]] <- c(psetCopy[[j]], set[i])
    }
  
  psetCopy <- c(psetCopy, set[i])
  
  pset <- c(pset, psetCopy)
  
}


#---- Sort the elements of the power list in terms of their cardinality ---------------------------#

psetSort <- vector(mode = "list", length = length(set))

# Create empty sublists which are matrices that have no. of columns equal to the cardinalities of
# the subsets
for (i in 1:(length(set))){
  
  psetSort[[i]] <- matrix(ncol = i)

}

# Assign the subsets to the corresponding matrix
for (i in 1:(length(pset))){
  
  cardi <- length(pset[[i]])
  
  psetSort[[cardi]] <- rbind(psetSort[[cardi]], as.vector(pset[[i]]))
  
}

# Delete the NA rows as placeholders
for (i in 1:(length(set))){
  
  psetSort[[i]] <- psetSort[[i]][-1,]
  
}

# Some werid technicality turned the matrix 1 into vector. Converting it back to matrix
psetSort[[1]] <- psetSort[[1]] %>% matrix(ncol = 1)

#---- Sort the elements of the power list in terms of their cardinality ---------------------------#

psetSort <- psetSort[-length(psetSort)]
probs <- rep(0, length(psetSort))

for (i in 1:length(psetSort)){
  
  probs[i] <- nrow(psetSort[[i]]) / (length(pset) - 1) # -1 because of deleting the improper subset
  
}


feasibleListElement <- list()
feasibleList <- list()
feasibleListLocation <- list()

iter = 1

for (iter in 1:1000){

  feasibleVector <- c()
  rest <- length(set) - length(feasibleVector)
  feasibleListElement <- list()  
  
  batches <- matrix(ncol = 2)
  colnames(batches) <- c("NumberOfOrders", "Index")
  
  while (rest > 0){
  
    # Pick a random number of batch (could be a batch with 1 order, up to how the maximum needed orders)
    repeat{
    
    pickedAmount <- sample(length(psetSort), 1, prob = probs, replace = T)
    
    if (pickedAmount <= rest){
      break
    }
    
    }
    
    # From that number of orders, pick an actual combination of orders 
    random <- sample(nrow(psetSort[[pickedAmount]]), 1)
    pickedOrder <- psetSort[[pickedAmount]][random,]
    
    # Append to feasible
    feasibleVector <- c(feasibleVector, pickedOrder)
    
    feasibleListElement <- c(feasibleListElement, list(pickedOrder))
    
    # Append the number of orders and the index to a matrix (for later retrival within the power set
    # list, so that we know exactly which subset we picked)
    batches <- rbind(batches, c(pickedAmount, random))
    
    # Recount the remaining spaces in the full order
    rest <- rest - length(feasibleVector)
    
  }
  
  print(batches)
  print(feasibleVector)
  
  # Remove NA placeholder row in object "batches"
  batches <- batches[-1,]
  
  if (isTRUE(feasibleVector == set) == TRUE){
    
    feasibleList <- c(feasibleList, list(feasibleListElement))
    feasibleListLocation <- c(feasibleListLocation, list(batches))
    
  }

}





for ( i in 1:length(pset)){
  
  if (length(pset[[i]]) > 4){
    pset[i] 
  }
  
}

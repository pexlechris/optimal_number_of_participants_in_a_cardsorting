import csv
import random
import Mantel # https://jwcarr.github.io/MantelTest/
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from clusim.clustering import Clustering
import clusim.sim as sim
import matplotlib.pyplot as plt

count_of_samples_for_each_n=10 #average of count_of_samples_for_each_n for n participants
perms_of_mantel_test=20#default is 10000




"""
#SKROUTZ
cards=54 #count of cards
total_participants=203 #count of all participants
#participants_range: lists with items the count of participants that we run the test
#so each of its items is a number from 0 to total_participants
#the 0 we use it only for the graphp O(0,0)
participants_range=[0,1,2,3,5,7,10,15,20,25,30,35,40,45,50,60]
column_category_label=3 #column 4 in csv (python starts from 0)

#column 1 of csv: participant_id
#column 2 of csv: card_index
#column 3 of csv: card_label
#column 4 of csv: category_label
#column 5 of csv: participant_sex
#column 6 of csv: participant_age
#column 7 of csv: participant_time_in_internet
#column 8 of csv: participant_previous_experience_at_eshop_domain
#column 9 of csv: difficult_cards
#column 10 of csv: how_dificult_was_the_procedure

#the row data of csv is sorted firstly by participant_id and secondly by card_index
all_data = []
with open(r"skroutz.csv", encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: # each row is a list
        all_data.append(row)
"""
        
        



#CELESTINO
cards=59 #count of cards
total_participants=210 #count of all participants
#participants_range: lists with items the count of participants that we run the test
#so each of its items is a number from 0 to total_participants
#the 0 we use it only for the graph O(0,0)
participants_range=[0,1,2,3,5,7,10,15,20,25,30,35,40,45,50,60]
column_category_label=3 #column 4 in csv (python starts from 0)

#column 1 of csv: participant_id
#column 2 of csv: card_index
#column 3 of csv: card_label
#column 4 of csv: category_label
#column 5 of csv: participant_sex
#column 6 of csv: participant_age
#column 7 of csv: participant_time_in_internet
#column 8 of csv: participant_previous_experience_at_eshop_domain
#column 9 of csv: difficult_cards
#column 10 of csv: how_dificult_was_the_procedure

#the row data of csv is sorted firstly by participant_id and secondly by card_index
all_data = []
with open(r"celestino.csv", encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: # each row is a list
        all_data.append(row)








def dissimilarity_matrix(some_participants):
    
    global column_category_label, all_data, cards, total_participants
    
    #participants_range: lists with items the count of participants that we run the test
    #so each of its items is a number from 0 to total_participants-1 (python starts counting from 0)
    #with random.sample we take UNIQUE numbers from the list in the first parameter of sample method
    #examples of random.sample() here: https://www.geeksforgeeks.org/python-random-sample-function
    participants_range=random.sample(range(0, total_participants), some_participants)
    
    #Initialize the dissimilarity matrix with zeros
    #The elements of the main diagonal will be always 0
    dissimilarity_matrix = [[0 for x in range(cards)] for y in range(cards)]
    
    for participant in participants_range:
        for x in range(participant*cards, (participant+1)*cards): #take cards/rows only form current participant
            
            for y in range(x+1,(participant+1)*cards):#Because the dissimilarity_matrix is symmetrical, we start the itteration form x+1 
                
                #if the cards have been sorted in differents groups from the current user
                if all_data[x][column_category_label]!=all_data[y][column_category_label]: 
                    
                    #We compare x-participant*cards and y-participant*cards in this itteration
                    # x-participant*cards is the card_index (0, 1, ... , cards-1) of the 1st compared card from participant
                    # y-participant*cards is the card_index (0, 1, ... , cards-1) of the 2nd compared card from participant                    
                    dissimilarity_matrix[x-participant*cards][y-participant*cards]=dissimilarity_matrix[x-participant*cards][y-participant*cards]+1
                    
                    #With the following line, we make the matrix symmetrical
                    dissimilarity_matrix[y-participant*cards][x-participant*cards]=dissimilarity_matrix[x-participant*cards][y-participant*cards]

    return dissimilarity_matrix





def clustering_with_clusim(dis):
    mat = np.array(dis)
    dists = squareform(mat)
    linkage_matrix = linkage(dists, "average")
    c = Clustering().from_scipy_linkage(linkage_matrix, dist_rescaled = True)
    return c






def mantel_elsim_r_average_and_errors(some_participants):
    
    global column_category_label, all_data, cards, total_participants, count_of_samples_for_each_n, perms_of_mantel_test
    dis2=dissimilarity_matrix(total_participants)
    
    c2 =clustering_with_clusim(dis2)    
    
    
    mantel_SUM=0
    elsim_SUM=0
    # r is between -1 and 1
    mantel_minimum=10 
    mantel_maximum=-10

    elsim_minimum=10 
    elsim_maximum=-10
    
    for i in range(count_of_samples_for_each_n):
        dis1=dissimilarity_matrix(some_participants)
        
        # Mantel Method
        mantel=Mantel.test(dis1, dis2, perms_of_mantel_test, method='pearson', tail='two-tail')
        mantel_r = mantel[0]
        
        #find errors (minimum and maximum)
        if mantel_r < mantel_minimum:
            mantel_minimum = mantel_r
        elif mantel_r > mantel_maximum:
            mantel_maximum = mantel_r
        mantel_SUM = mantel_SUM + mantel_r


        c1 =clustering_with_clusim(dis1)
        
        # Element-centric Similarity
        elsim_r = sim.element_sim(c1, c2, r=1.0, alpha=0.9)

        #find errors (minimum and maximum)
        if elsim_r < elsim_minimum:
            elsim_minimum = elsim_r
        elif elsim_r > elsim_maximum:
            elsim_maximum = elsim_r
        elsim_SUM = elsim_SUM + elsim_r
        
        
    mantel_average = mantel_SUM / count_of_samples_for_each_n #average of mantel_r
    mantel_l_error = mantel_average - mantel_minimum #mantel_lower_error
    mantel_u_error = mantel_maximum - mantel_average #mantel_upper_error

    elsim_average = elsim_SUM / count_of_samples_for_each_n #average of elsim_r
    elsim_l_error = elsim_average - elsim_minimum #mantel_lower_error
    elsim_u_error = elsim_maximum - elsim_average #mantel_upper_error
    
    return mantel_average, mantel_l_error, mantel_u_error, elsim_average, elsim_l_error, elsim_u_error






def mantel_elsim_r_average_and_errors_in_participants_range(participants_range):
    
    global total_participants
    
    #Initialization with zeros
    
    mantel_r_average_of_each_n=[]
    mantel_r_lower_error_of_each_n=[]
    mantel_r_upper_error_of_each_n=[]
    
    elsim_r_average_of_each_n=[]
    elsim_r_lower_error_of_each_n=[]
    elsim_r_upper_error_of_each_n=[]
    
    for y in range(0,total_participants+1):
        mantel_r_average_of_each_n.append(0)
        mantel_r_lower_error_of_each_n.append(0)
        mantel_r_upper_error_of_each_n.append(0)
        
        elsim_r_average_of_each_n.append(0)
        elsim_r_lower_error_of_each_n.append(0)
        elsim_r_upper_error_of_each_n.append(0)
        
        
        
        
    # x is the number/id of participant (possible values: 1 , ... , total_participants)
    for x in participants_range:
        if x==0: 
            #0 ~ O(0,0) is the axis origin
            mantel_r_average_of_each_n[0]=0
            mantel_r_lower_error_of_each_n[0]=0
            mantel_r_upper_error_of_each_n[0]=0
            
            elsim_r_average_of_each_n[0]=0
            elsim_r_lower_error_of_each_n[0]=0
            elsim_r_upper_error_of_each_n[0]=0
        else:
            mantel_average, mantel_l_error, mantel_u_error, elsim_average, elsim_l_error, elsim_u_error = mantel_elsim_r_average_and_errors(x)
            
            #put the values in an array
            mantel_r_average_of_each_n[x]=mantel_average
            mantel_r_lower_error_of_each_n[x]=mantel_l_error
            mantel_r_upper_error_of_each_n[x]=mantel_u_error

            elsim_r_average_of_each_n[x]=elsim_average
            elsim_r_lower_error_of_each_n[x]=elsim_l_error
            elsim_r_upper_error_of_each_n[x]=elsim_u_error
            
    #we return the arrays
    return mantel_r_average_of_each_n, mantel_r_lower_error_of_each_n, mantel_r_upper_error_of_each_n, elsim_r_average_of_each_n, elsim_r_lower_error_of_each_n, elsim_r_upper_error_of_each_n






def save_errorbar(r_average_of_each_n, r_lower_error_of_each_n, r_upper_error_of_each_n,  participants_range, title, xlabel, ylabel, save, clear):
    global total_participants
    
    x=[np.array(range(0,total_participants+1))[i] for i in participants_range] 
    y=[r_average_of_each_n[i] for i in participants_range]
    
    lower_error = [r_lower_error_of_each_n[i] for i in participants_range] 
    upper_error = [r_upper_error_of_each_n[i] for i in participants_range]
    asymmetric_error = [lower_error, upper_error]
    
    plt.errorbar(x, y, yerr=asymmetric_error)
    plt.title(title)
    plt.xlabel(xlabel)

    plt.ylabel(ylabel)
    if save==True:
        plt.savefig(title+".png",dpi=300)
    if clear==True:
        plt.show()




mantel_r_average_of_each_n, mantel_r_lower_error_of_each_n, mantel_r_upper_error_of_each_n, elsim_r_average_of_each_n, elsim_r_lower_error_of_each_n, elsim_r_upper_error_of_each_n = mantel_elsim_r_average_and_errors_in_participants_range(participants_range)

i=0
for i in participants_range:
    if i==0:
        print("participants;", " ", "mantel;", "", "elsim")
        continue
    print(i,"; ", mantel_r_average_of_each_n[i], "; ", elsim_r_average_of_each_n[i]) 


#CASE: 2 graphs together
save_errorbar(mantel_r_average_of_each_n, mantel_r_lower_error_of_each_n, mantel_r_upper_error_of_each_n, participants_range, "Mantel", "Sample Size", "Average correlation", False, False)
save_errorbar(elsim_r_average_of_each_n, elsim_r_lower_error_of_each_n, elsim_r_upper_error_of_each_n, participants_range, "Elsim & Mantel Error Bar", "Sample Size", "Average correlation", True, True)

#CASE: graphs seperately
save_errorbar(mantel_r_average_of_each_n, mantel_r_lower_error_of_each_n, mantel_r_upper_error_of_each_n, participants_range, "Mantel Error Bar", "Sample Size", "Average correlation", True, True)
save_errorbar(elsim_r_average_of_each_n, elsim_r_lower_error_of_each_n, elsim_r_upper_error_of_each_n, participants_range, "Elsim Error Bar", "Sample Size", "Average correlation", True, True)



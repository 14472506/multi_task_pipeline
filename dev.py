#a = 174
#b = 156
#
#even = int(a/b)
#rem = a%b
#
#print(even)
#print(rem)
#
#eve_dist = a + b - rem
#
#rem_even = int(eve_dist/rem)
#rem_rem = eve_dist%rem
#
#print(rem_even)
#print(rem_rem)
#
#eve_dist_2 = eve_dist + rem_even - rem_rem
#
#rem_even_2 = int(eve_dist_2/rem_rem)
#rem_rem_2 = eve_dist%rem_rem
#
#print(rem_even_2)
#print(rem_rem_2)

def multi_task_training_scheduler(max_list, min_list):
    """
    something
    """
    if len(min_list) == 0:
        return(max_list)
    else:
        even = int(len(max_list)/len(min_list))
        rem = len(max_list)%len(min_list)
        iters = len(max_list) - rem
        print(rem)
        new_max = []
        min_count = 0
        for i in range(iters):
            if i%even == 0:
                new_max.append(min_list[min_count])
                min_count += 1
            new_max.append(max_list[i])
        if rem == 0:
            new_min = []
        else:
            new_min = max_list[-rem:]
        return multi_task_training_scheduler(new_max, new_min)

max_list = ["a"]*1074
min_list = ["b"]*156

output = fuck_this(max_list, min_list)
print("done")
print(output)
print(len(output))
print(len([x for x in output if x == "a"]))
print(len([x for x in output if x == "b"]))


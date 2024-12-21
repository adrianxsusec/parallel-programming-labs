import random
from time import sleep
from mpi4py import MPI

tab = '\t'

MY_LEFT, MY_RIGHT = 0, 1
HIS_LEFT, HIS_RIGHT = 0, 1

HUNGRY, CAN_EAT, THINKING = 0, 1, 2
SEND_FORK, REQUEST_FORK = 0, 1

MIN_EATING_TIME = 1
MAX_EATING_TIME = 10

MIN_THINKING_TIME = 1
MAX_THINKING_TIME = 10

comm = MPI.COMM_WORLD
status = MPI.Status()
rank = comm.Get_rank()
size = comm.Get_size()

# works
left_neighbor = (rank - 1 + size) % size
right_neighbor = (rank + 1) % size

state = THINKING

if rank == 0:
    right_fork, left_fork = True, True
    right_dirty, left_dirty = True, True
elif rank == size - 1:
    right_fork, left_fork = False, False
    right_dirty, left_dirty = False, False
else:
    right_fork, left_fork = True, False
    right_dirty, left_dirty = True, False
    
requested_right, requested_left = False, False
received_right_request, received_left_request = False, False


def tabbed_print(msg):
    print(f"{tab * rank} {msg}")


def remember_request(requester, fork):
    global requested_right, requested_left
    if requester == left_neighbor and fork == HIS_RIGHT:
        requested_left = True
    elif requester == right_neighbor and fork == HIS_LEFT:
        requested_right = True
    else:
        raise ValueError(f"Unknown requester: {requester}")
    
    
def can_send_left_fork():
    global left_fork, left_dirty
    return left_fork and left_dirty


def can_send_right_fork():
    global right_fork, right_dirty
    return right_fork and right_dirty
    

def make_forks_dirty():
    global right_dirty, left_dirty
    right_dirty = True
    left_dirty = True
    

def dispatch_left_fork():
    global left_fork, left_dirty, received_left_request
    comm.send(MY_LEFT, dest=left_neighbor, tag=SEND_FORK)
    left_fork = False
    left_dirty = False
    received_left_request = False
    # tabbed_print(f"Sending left fork to {left_neighbor}, now i have left fork {left_fork} and right fork {right_fork}")
    
    
def dispatch_right_fork():
    global right_fork, right_dirty, received_right_request
    comm.send(MY_RIGHT, dest=right_neighbor, tag=SEND_FORK)
    right_fork = False
    right_dirty = False
    received_right_request = False
    # tabbed_print(f"Sending right fork to {right_neighbor}, now i have left fork {left_fork} and right fork {right_fork}")


def fulfill_remembered_requests_if_existing():
    global requested_right, requested_left
    if requested_left and can_send_left_fork():
        dispatch_left_fork()
        
    if requested_right and can_send_right_fork():
        dispatch_right_fork()


def probe_requests():
    global received_right_request, received_left_request
    
    fulfill_remembered_requests_if_existing()
        
    has_request = comm.iprobe(source=MPI.ANY_SOURCE, tag=REQUEST_FORK, status=status)
    
    # currently received request
    if has_request:
        buf = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        requester = status.source
        if requester == left_neighbor and buf == HIS_RIGHT:
            if can_send_left_fork():
                dispatch_left_fork()
            else:
                remember_request(requester, buf)
            
        elif requester == right_neighbor and buf == HIS_LEFT:
            if can_send_right_fork():
                dispatch_right_fork()
            else:
                remember_request(requester, buf)
                
                
def request_right_fork():
    global requested_right
    comm.send(MY_RIGHT, dest=right_neighbor, tag=REQUEST_FORK)
    requested_right = True
    # tabbed_print(f"Requesting right fork from {right_neighbor}, now i have left fork {left_fork} and right fork {right_fork}")
    

def request_left_fork():
    global requested_left
    comm.send(MY_LEFT, dest=left_neighbor, tag=REQUEST_FORK)
    requested_left = True
    # tabbed_print(f"Requesting left fork from {left_neighbor}, now i have left fork {left_fork} and right fork {right_fork}")
                
                
def request_missing_forks():
    if not left_fork and not requested_left:
        request_left_fork()
        
    if not right_fork and not requested_right:
        request_right_fork()
        

def think_and_fulfill_requests():
    global state
    tabbed_print("Thinking")
    thinking_time = random.randint(MIN_THINKING_TIME, MAX_THINKING_TIME)
    for _ in range(thinking_time):
        probe_requests()
        sleep(1)
    # tabbed_print(f"Done thinking, now I'm hungry, left fork {left_fork}, right fork {right_fork}")
    state = HUNGRY
    
    
def requested_fork_missing():
    global requested_left, requested_right
    return (requested_left and not left_fork) or (requested_right and not right_fork)


def accept_left_fork():
    global left_fork, left_dirty
    left_fork = True
    left_dirty = False
    # tabbed_print(f"Accepted left fork from {left_neighbor}, now i have left fork {left_fork} and right fork {right_fork}")
    
    
def accept_right_fork():
    global right_fork, right_dirty
    right_fork = True
    right_dirty = False
    # tabbed_print(f"Accepted right fork from {right_neighbor}, now i have left fork {left_fork} and right fork {right_fork}")
    
    
def hungry():
    global state, left_fork, right_fork
    while not (left_fork and right_fork):
        request_missing_forks()
        fulfill_remembered_requests_if_existing()
        
        while requested_fork_missing():
            buf = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.tag == SEND_FORK:
                sender = status.source
                if sender == left_neighbor and buf == HIS_RIGHT:
                    accept_left_fork()
                elif sender == right_neighbor and buf == HIS_LEFT:
                    accept_right_fork()
                    
            elif status.tag == REQUEST_FORK:
                requester = status.source
                if requester == left_neighbor and can_send_left_fork() and buf == HIS_LEFT:
                    dispatch_left_fork()
                elif requester == right_neighbor and can_send_right_fork() and buf == HIS_RIGHT:
                    dispatch_right_fork()
                else:
                    remember_request(requester, buf)
    
    state = CAN_EAT


def eat():
    global state, left_dirty, right_dirty
    tabbed_print(f"Eating, left fork {left_fork}, right fork {right_fork}")
    sleep(random.randint(MIN_EATING_TIME, MAX_EATING_TIME))
    make_forks_dirty()
    state = THINKING


def main():
    while True:
        if state == THINKING:
            think_and_fulfill_requests()
        elif state == HUNGRY:
            hungry()
        elif state == CAN_EAT:
            eat()
        else:
            raise ValueError(f"Unknown state: {state}")
        
        
main()
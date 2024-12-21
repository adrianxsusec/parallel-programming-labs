import random
import time

from board import Board
from mpi4py import MPI

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
WINNING_LENGTH = 4

MASTER_DEPTH = 3
MAX_DEPTH = 8 - MASTER_DEPTH

CPU = 1
HUMAN = 2

WORK_TAG = 0
COMPLETED_TAG = 1

mpi_comm = MPI.COMM_WORLD
mpi_status = MPI.Status()
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


class Task:
    def __init__(self, id: int, last_col_played: int, current_player: int, board: Board):
        self.id = id
        self.last_col_played = last_col_played
        self.current_player = current_player
        self.board = board


class Result:
    def __init__(self, id: int, value: float):
        self.id = id
        self.value = value

    def __str__(self):
        return f"Task ID: {self.id}, Value: {self.value}"


def get_opponent(current_player: int) -> int:
    if current_player == HUMAN:
        return CPU
    else:
        return HUMAN


def create_tasks(board: Board, depth: int):
    current_player = CPU

    tasks = [Task(0, 0, current_player, board)]

    for level in range(depth):
        level_tasks = []

        for idx, task in enumerate(tasks):
            for col in range(BOARD_WIDTH):
                new_board = task.board.copy()
                if not new_board.move_legal(col):
                    continue

                task_id = idx * 7 + col

                new_board.make_move(col, current_player)
                level_tasks.append(Task(task_id, col, get_opponent(current_player), new_board))

        tasks = level_tasks

        current_player = HUMAN if current_player == CPU else CPU

    return tasks


def process_task(task: Task, max_depth: int):
    def dfs(board: Board, last_col_played: int, current_player: int, remaining_depth: int) -> float:
        if remaining_depth == 0:
            return 0

        end = board.game_end(last_col_played)

        if end[0]:
            if end[1] == CPU:
                return 1
            else:
                return -1

        results = []

        for col in range(0, BOARD_WIDTH):
            new_board = board.copy()
            if not new_board.move_legal(col):
                continue

            new_board.make_move(col, current_player)

            result = dfs(new_board, col, get_opponent(current_player), remaining_depth - 1)
            results.append(result)

        if current_player == CPU and 1 in results:
            return 1
        elif current_player == HUMAN and -1 in results:
            return -1
        elif all(res == -1 for res in results):
            return -1
        elif all(res == 1 for res in results):
            return 1
        else:
            return sum(results) / BOARD_WIDTH

    result = dfs(task.board, task.last_col_played, task.current_player, max_depth)

    return Result(task.id, result)


def process_results(results: list[Result], master_depth: int, branching_factor: int = 7):
    sorted_results = sorted(results, key=lambda x: x.id, reverse=False)
    [print(f"Task ID: {task.id}, Value: {task.value}") for task in sorted_results]

    # keep in mind that there won't always be 49 or 343 tasks - when a move is impossible, the task is not generated
    # BUT, the id of the task is still equal to its "position" in the search tree

    current_level = master_depth

    if current_level % 2 == 1:
        current_player = CPU
    else:
        current_player = HUMAN

    while len(sorted_results) != 7:
        expected_tasks = pow(int(branching_factor), int(current_level))

        level_results = []

        for i in range(0, expected_tasks, branching_factor):
            min_id = i
            max_id = i + branching_factor
            subtasks = []
            game_ender = 0

            for result in filter(lambda x: min_id <= x.id < max_id, sorted_results):
                if result.value == -1 or result.value == 1:
                    game_ender = result.value
                subtasks.append(result.value)

            res_id = int(i / branching_factor)

            if current_player == HUMAN and game_ender == -1:
                level_results.append(Result(res_id, -1))

            elif current_player == CPU and game_ender == 1:
                level_results.append(Result(res_id, 1))
            else:
                res = sum(subtasks) / len(subtasks) if subtasks else 0
                level_results.append(Result(res_id, res))

        sorted_results = level_results

        current_level -= 1

    best_col = sorted_results.index(max(sorted_results, key=lambda x: x.value))

    return best_col


if __name__ == '__main__':
    board = Board(BOARD_WIDTH, BOARD_HEIGHT)

    CURR_MOVE = random.randint(0, 1)

    GAME_FINISHED = 0

    # master
    if mpi_rank == 0:
        print("Starting game - HUMAN: O, CPU: X")
        while not GAME_FINISHED:
            if CURR_MOVE == HUMAN:
                print(board)
                print("Your turn, select a column:")

                input_accepted = False

                user_column = None

                while not input_accepted:
                    try:
                        user_column = input()
                        if len(str(user_column)) > 1:
                            raise ValueError

                        user_column = int(user_column)

                        if not board.move_legal(user_column):
                            raise ValueError

                        print(str(user_column))
                        input_accepted = True
                        board.make_move(user_column, player=CURR_MOVE)

                    except ValueError:
                        print("Enter a valid number and try again:")

                if board.game_end(user_column)[0]:
                    print("Human won!")
                    GAME_FINISHED = 1
                else:
                    CURR_MOVE = CPU

            else:
                start_time = time.time()
                tasks = create_tasks(board, MASTER_DEPTH)
                expected_completed_tasks = len(tasks)
                completed_tasks = []

                # send initial tasks
                for worker in range(1, mpi_size):
                    if tasks:
                        task = tasks.pop(0)
                        mpi_comm.send(obj=task, dest=worker, tag=WORK_TAG)

                # receive from workers and send remaining tasks
                while tasks:
                    result = mpi_comm.recv(source=MPI.ANY_SOURCE, status=mpi_status, tag=COMPLETED_TAG)
                    completed_tasks.append(result)
                    new_task = tasks.pop(0)
                    mpi_comm.send(obj=new_task, dest=mpi_status.source, tag=WORK_TAG)

                # collect remaining tasks
                while len(completed_tasks) != expected_completed_tasks:
                    result = mpi_comm.recv(source=MPI.ANY_SOURCE, tag=COMPLETED_TAG)
                    completed_tasks.append(result)

                print("Received all tasks")

                best_move = process_results(completed_tasks, MASTER_DEPTH)

                end_time = time.time()
                print(f"Time elapsed: {end_time - start_time}")

                if board.move_legal(best_move):
                    board.make_move(best_move, player=CURR_MOVE)

                if board.game_end(best_move)[0]:
                    print("CPU won!")
                    GAME_FINISHED = 1
                else:
                    CURR_MOVE = HUMAN

        print(board)

        for worker in range(1, mpi_size):
            mpi_comm.send(obj="kill", dest=worker, tag=WORK_TAG)

    # workers
    else:
        while True:
            task = mpi_comm.recv(source=0, tag=WORK_TAG)

            if task == "kill":
                break

            result = process_task(task, MAX_DEPTH)
            mpi_comm.send(obj=result, dest=0, tag=COMPLETED_TAG)

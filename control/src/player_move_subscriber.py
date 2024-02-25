import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import random

# Function to draw a 3x3 grid on the image and number the cells
def draw_3x3_grid(image, contour):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the width and height of each cell
    cell_width = w // 3
    cell_height = h // 3
    
    # Draw horizontal lines
    for i in range(1, 3):
        cv2.line(image, (x, y + i * cell_height), (x + w, y + i * cell_height), (0, 255, 0), 2)
    
    # Draw vertical lines
    for i in range(1, 3):
        cv2.line(image, (x + i * cell_width, y), (x + i * cell_width, y + h), (0, 255, 0), 2)
    
    # Number the cells from top-left corner
    cell_number = 1
    for i in range(3):
        for j in range(3):
            # Calculate the position for text label
            text_x = x + j * cell_width + cell_width // 2 - 10
            text_y = y + i * cell_height + cell_height // 2 + 10
            # Add cell number
            # Increment cell number
            cell_number += 1

# Function to check if a player has won
def check_winner(board, player):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] == player:
            return True
        if board[0][i] == board[1][i] == board[2][i] == player:
            return True
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

class Tic(object):

    winning_combos = (
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6])
    winners = ('X-win', 'Draw', 'O-win')

    def __init__(self, squares=[]):
        """Initialize either custom or default board"""
        if len(squares) == 0:
            self.squares = [None for i in range(9)]
        else:
            self.squares = squares

    def show(self):
        """Print game progress"""
        for element in [
                self.squares[i: i + 3] for i in range(0, len(self.squares), 3)]:
            print(element)

    def available_moves(self):
        return [k for k, v in enumerate(self.squares) if v is None]

    def available_combos(self, player):
        return self.available_moves() + self.get_squares(player)

    def complete(self):
        """Check if game has ended"""
        if None not in [v for v in self.squares]:
            return True
        if self.winner() is not None:
            return True
        return False

    def X_won(self):
        return self.winner() == 'X'

    def O_won(self):
        return self.winner() == 'O'

    def tied(self):
        return self.complete() and self.winner() is None

    def winner(self):
        for player in ('X', 'O'):
            positions = self.get_squares(player)
            for combo in self.winning_combos:
                win = True
                for pos in combo:
                    if pos not in positions:
                        win = False
                if win:
                    return player
        return None

    def get_squares(self, player):
        """Returns squares belonging to a player"""
        return [k for k, v in enumerate(self.squares) if v == player]

    def make_move(self, position, player):
        self.squares[position] = player

    def alphabeta(self, node, player, alpha, beta):
        """Alphabeta algorithm"""
        if node.complete():
            if node.X_won():
                return -1
            elif node.tied():
                return 0
            elif node.O_won():
                return 1

        for move in node.available_moves():
            node.make_move(move, player)
            val = self.alphabeta(node, get_enemy(player), alpha, beta)
            node.make_move(move, None)
            if player == 'O':
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    return beta
            else:
                if val < beta:
                    beta = val
                if beta <= alpha:
                    return alpha
        return alpha if player == 'O' else beta


def get_enemy(player):
    if player == 'X':
        return 'O'
    return 'X'


def determine(board, player):
    """Determine best possible move"""
    a = -2
    choices = []
    if len(board.available_moves()) == 9:
        return 4
    for move in board.available_moves():
        board.make_move(move, player)
        val = board.alphabeta(board, get_enemy(player), -2, 2)
        board.make_move(move, None)
        if val > a:
            a = val
            choices = [move]
        elif val == a:
            choices.append(move)
    return random.choice(choices)


class TicTacToeSubscriberNode(Node):
    def __init__(self):
        super().__init__('tic_tac_toe_subscriber_node')
        self.subscription = self.create_subscription(
            Image,
            'webcam_topic',
            self.webcam_callback,
            10)
        self.subscription  # Prevent unused variable warning
        self.board = Tic()
        self.player_turn = True

    def webcam_callback(self, msg):
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize the frame to half its original size
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of red color in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        # Iterate through the contours and find the centroid of the largest contour
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate centroid using image moments
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Iterate through the cells of the 3x3 grid
                for i in range(3):
                    for j in range(3):
                        # Calculate the top-left corner coordinates of the cell
                        cell_x = j * (frame.shape[1] // 3)
                        cell_y = i * (frame.shape[0] // 3)
                        # Calculate the bottom-right corner coordinates of the cell
                        cell_x_end = (j + 1) * (frame.shape[1] // 3)
                        cell_y_end = (i + 1) * (frame.shape[0] // 3)
                        # Check if the centroid falls within the current cell
                        if cell_x < cx < cell_x_end and cell_y < cy < cell_y_end:
                            # Check if the cell is empty
                            if self.board.squares[i * 3 + j] is None and self.player_turn:
                                self.board.squares[i * 3 + j] = 'X'  # Player's move
                                self.player_turn = False
                            elif self.board.squares[i * 3 + j] is None and not self.player_turn:
                                move = determine(self.board, 'O')
                                if self.board.squares[move] is None:
                                    self.board.squares[move] = 'O'  # AI's move
                                    self.player_turn = True
        # Draw player's moves (red circles) and AI's moves (blue circles)
        for i in range(3):
            for j in range(3):
                if self.board.squares[i * 3 + j] == 'X':
                    cell_x = j * (frame.shape[1] // 3) + (frame.shape[1] // 6)
                    cell_y = i * (frame.shape[0] // 3) + (frame.shape[0] // 6)
                    cv2.circle(frame, (cell_x, cell_y), 30, (0, 0, 255), 3)  # Player's move (red circle)
                elif self.board.squares[i * 3 + j] == 'O':
                    cell_x = j * (frame.shape[1] // 3) + (frame.shape[1] // 6)
                    cell_y = i * (frame.shape[0] // 3) + (frame.shape[0] // 6)
                    cv2.circle(frame, (cell_x, cell_y), 30, (255, 0, 0), 3)  # AI's move (blue circle)

        # Display the frame
        cv2.imshow("Tic Tac Toe", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    tic_tac_toe_subscriber_node = TicTacToeSubscriberNode()
    rclpy.spin(tic_tac_toe_subscriber_node)

if __name__ == '__main__':
    main()

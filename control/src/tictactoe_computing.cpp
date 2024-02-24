#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

class Tic {
private:
    std::vector<int> squares;
    const std::vector<std::vector<int>> winning_combos = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    };
public:
    Tic() : squares(9, -1) {} // Initialize squares to -1
    
    void show() {
        for (size_t i = 0; i < 9; i += 3) {
            for (size_t j = 0; j < 3; ++j) {
                std::cout << squares[i + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<int> available_moves() {
        std::vector<int> moves;
        for (size_t i = 0; i < squares.size(); ++i) {
            if (squares[i] == -1) {
                moves.push_back(i);
            }
        }
        return moves;
    }

    std::vector<int> get_squares(int player) {
        std::vector<int> player_squares;
        for (size_t i = 0; i < squares.size(); ++i) {
            if (squares[i] == player) {
                player_squares.push_back(i);
            }
        }
        return player_squares;
    }

    bool complete() {
        if (std::find(squares.begin(), squares.end(), -1) == squares.end()) {
            return true; // All squares filled
        }
        return winner() != -1;
    }

    int winner() {
        for (int player = 0; player <= 1; ++player) {
            std::vector<int> positions = get_squares(player);
            for (const auto& combo : winning_combos) {
                bool win = true;
                for (int pos : combo) {
                    if (std::find(positions.begin(), positions.end(), pos) == positions.end()) {
                        win = false;
                        break;
                    }
                }
                if (win) return player;
            }
        }
        return -1; // No winner yet
    }

    void make_move(int position, int player) {
        squares[position] = player;
    }

    int alphabeta(Tic& node, int player, int alpha, int beta) {
        if (node.complete()) {
            if (node.winner() == 1) return -1;
            else if (node.winner() == 0) return 0;
            else if (node.winner() == 2) return 1;
        }

        for (int move : node.available_moves()) {
            node.make_move(move, player);
            int val = alphabeta(node, get_enemy(player), alpha, beta);
            node.make_move(move, -1);
            if (player == 1) {
                alpha = std::max(alpha, val);
                if (alpha >= beta) return beta;
            } else {
                beta = std::min(beta, val);
                if (beta <= alpha) return alpha;
            }
        }
        return (player == 1) ? alpha : beta;
    }

    int determine(int player) {
        int a = -2;
        std::vector<int> choices;
        if (available_moves().size() == 9) return 4;
        for (int move : available_moves()) {
            make_move(move, player);
            int val = alphabeta(*this, get_enemy(player), -2, 2);
            make_move(move, -1);
            if (val > a) {
                a = val;
                choices = {move};
            } else if (val == a) {
                choices.push_back(move);
            }
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, choices.size() - 1);
        return choices[dis(gen)];
    }

    int get_enemy(int player) {
        return (player == 0) ? 1 : 0;
    }
};

void draw_3x3_grid(cv::Mat& image, std::vector<cv::Point>& contour) {
    cv::Rect boundingRect = cv::boundingRect(contour);

    int cell_width = boundingRect.width / 3;
    int cell_height = boundingRect.height / 3;

    for (int i = 1; i < 3; ++i) {
        cv::line(image, cv::Point(boundingRect.x, boundingRect.y + i * cell_height),
                 cv::Point(boundingRect.x + boundingRect.width, boundingRect.y + i * cell_height),
                 cv::Scalar(0, 255, 0), 2);
    }

    for (int i = 1; i < 3; ++i) {
        cv::line(image, cv::Point(boundingRect.x + i * cell_width, boundingRect.y),
                 cv::Point(boundingRect.x + i * cell_width, boundingRect.y + boundingRect.height),
                 cv::Scalar(0, 255, 0), 2);
    }

    int cell_number = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int text_x = boundingRect.x + j * cell_width + cell_width / 2 - 10;
            int text_y = boundingRect.y + i * cell_height + cell_height / 2 + 10;
            cv::putText(image, std::to_string(cell_number), cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cell_number++;
        }
    }
}

int main() {
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    Tic board;
    bool player_turn = true;

    while (true) {
        cv::Mat frame;
        cap.read(frame);

        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        cv::Mat gray, blur, thresh;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(blur, thresh, 255, 1, 1, 19, 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point>> filtered_contours;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 1900) {
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, 0.1 * cv::arcLength(contour, true), true);
                if (approx.size() == 4) {
                    filtered_contours.push_back(contour);
                    draw_3x3_grid(frame, contour);
                }
            }
        }

        if (player_turn) {
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::Mat mask;
            cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                std::vector<cv::Point> largest_contour = *std::max_element(contours.begin(), contours.end(),
                    [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                        return cv::contourArea(a) < cv::contourArea(b);
                    });

                cv::Moments M = cv::moments(largest_contour);
                if (M.m00 != 0) {
                    int cx = static_cast<int>(M.m10 / M.m00);
                    int cy = static_cast<int>(M.m01 / M.m00);
                    int cell_row = cy / (frame.rows / 3);
                    int cell_col = cx / (frame.cols / 3);
                    int move = cell_row * 3 + cell_col;
                    if (move >= 0 && move < 9 && board.available_moves()[move]) {
                        board.make_move(move, 0);
                        player_turn = false;
                    }
                }
            }
        } else {
            int move = board.determine(1);
            if (board.available_moves()[move]) {
                board.make_move(move, 1);
                player_turn = true;
            }
        }

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (board.available_moves()[i * 3 + j] == 0) {
                    int cell_x = j * (frame.cols / 3) + (frame.cols / 6);
                    int cell_y = i * (frame.rows / 3) + (frame.rows / 6);
                    cv::circle(frame, cv::Point(cell_x, cell_y), 30, cv::Scalar(0, 0, 255), 3);
                } else if (board.available_moves()[i * 3 + j] == 1) {
                    int cell_x = j * (frame.cols / 3) + (frame.cols / 6);
                    int cell_y = i * (frame.rows / 3) + (frame.rows / 6);
                    cv::circle(frame, cv::Point(cell_x, cell_y), 30, cv::Scalar(255, 0, 0), 3);
                }
            }
        }

        if (board.complete()) {
            int winner = board.winner();
            if (winner == 0) {
                cv::putText(frame, "Player wins!", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            } else if (winner == 1) {
                cv::putText(frame, "AI wins!", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            } else {
                cv::putText(frame, "It's a draw!", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
            }
            break;
        }

        cv::imshow("Tic Tac Toe", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

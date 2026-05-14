"""
Pygame 可视化模块 — 零外部依赖（仅需 pygame + numpy）
展示训练好的 Q-learning 智能体解迷宫
"""


def run_visualization(trainer, fps=10):
    """Launch pygame to visualize a trained agent solving the maze."""
    try:
        import pygame
    except ImportError:
        print("pygame 未安装。请执行: pip install pygame")
        return

    WINDOW_SIZE = 800
    maze = trainer.maze
    CELL_SIZE = WINDOW_SIZE // (maze.size + 2)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (173, 216, 230)
    GRAY = (128, 128, 128)
    YELLOW = (255, 255, 0)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE + 200, WINDOW_SIZE))
    pygame.display.set_caption("Q-Learning Maze Solver (Improved)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 14)

    agent_x, agent_y = maze.start
    path = [(agent_x, agent_y)]
    step = 0
    done = False
    paused = False

    def draw():
        screen.fill(WHITE)
        for x in range(maze.size):
            for y in range(maze.size):
                rect = pygame.Rect(
                    CELL_SIZE + y * CELL_SIZE, CELL_SIZE + x * CELL_SIZE,
                    CELL_SIZE - 1, CELL_SIZE - 1)
                if maze.maze[x][y] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                elif (x, y) == maze.start:
                    pygame.draw.rect(screen, GREEN, rect)
                elif (x, y) == maze.end:
                    pygame.draw.rect(screen, RED, rect)

        for (px, py) in path:
            rect = pygame.Rect(
                CELL_SIZE + py * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE + px * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=5)

        ax = CELL_SIZE + agent_y * CELL_SIZE + CELL_SIZE // 2
        ay = CELL_SIZE + agent_x * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, BLUE, (ax, ay), CELL_SIZE // 3)

        # Right-side info panel
        pygame.draw.rect(screen, GRAY, pygame.Rect(WINDOW_SIZE, 0, 200, WINDOW_SIZE))
        info_lines = [
            f"Step: {step}",
            f"Pos: ({agent_x},{agent_y})",
            f"Goal: {maze.end}",
            "",
            "Keys:",
            "R: Reset",
            "P: Pause",
            "Q: Quit",
        ]
        y = 20
        for line in info_lines:
            surf = small_font.render(line, True, WHITE)
            screen.blit(surf, (WINDOW_SIZE + 10, y))
            y += 25

        if done:
            surf = font.render("GOAL REACHED!", True, GREEN)
            screen.blit(surf, (WINDOW_SIZE + 10, WINDOW_SIZE - 60))
        elif paused:
            surf = font.render("PAUSED", True, YELLOW)
            screen.blit(surf, (WINDOW_SIZE + 10, WINDOW_SIZE - 60))

        pygame.display.update()

    # Set agent to eval mode (no exploration)
    trainer.agent.set_eval_mode()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    agent_x, agent_y = maze.start
                    path = [(agent_x, agent_y)]
                    step = 0
                    done = False
                elif event.key == pygame.K_p:
                    paused = not paused

        if not done and not paused:
            state = maze.get_state_index(agent_x, agent_y)
            valid = maze.get_valid_actions(agent_x, agent_y)
            action, _ = trainer.agent.choose_action(state, valid)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = moves[action]
            nx, ny = agent_x + dx, agent_y + dy
            if maze.is_valid(nx, ny):
                agent_x, agent_y = nx, ny
                path.append((nx, ny))
            step += 1
            if (agent_x, agent_y) == maze.end:
                done = True

        draw()
        clock.tick(fps)

    pygame.quit()

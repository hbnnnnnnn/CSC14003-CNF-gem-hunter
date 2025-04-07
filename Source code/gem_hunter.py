import pygame
import random
from main import read_matrix, prepare_cnf_data, solve_cnf_pysat, solve_cnf_backtracking, solve_cnf_brute_force, interpret_model
pygame.init()
pygame.mixer.init()
normal_font = pygame.font.Font('Source code/assets/text/Emulogic-zrEw.ttf', 12)
header_font = pygame.font.Font('Source code/assets/text/Emulogic-zrEw.ttf', 32)
small_font = pygame.font.Font('Source code/assets/text/Emulogic-zrEw.ttf', 10)

class Button:
    def __init__(self, x, y, width, height, text, color=pygame.Color('purple'), hover_color=pygame.Color('lavender'), text_color=pygame.Color('white')):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        
    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, 'black', self.rect, 2)
        
        text_surface = normal_font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered
        
    def is_clicked(self, mouse_pos, mouse_click):
        return self.rect.collidepoint(mouse_pos) and mouse_click

class Cell:
    def __init__(self, row, col, size, value):
        self.row = row
        self.col = col
        self.size = size
        self.value = value
        self.revealed = False
        self.flagged = False
        self.rect = pygame.Rect(col * size, row * size, size, size)

    def draw(self, surface, offset_x, offset_y):
        draw_rect = pygame.Rect(
            self.rect.x + offset_x,
            self.rect.y + offset_y,
            self.rect.width,
            self.rect.height
        )
       
        if not self.revealed:
            if self.flagged:
                pygame.draw.rect(surface, 'yellow', draw_rect)
                flag_font = pygame.font.SysFont('Arial', int(self.size * 0.7))
                flag_text = flag_font.render('F', True, 'red')
                flag_rect = flag_text.get_rect(center=(
                    draw_rect.x + draw_rect.width // 2,
                    draw_rect.y + draw_rect.height // 2
                ))
                surface.blit(flag_text, flag_rect)
            else:
                pygame.draw.rect(surface, 'darkgray', draw_rect)
        else:
            if self.value == 'G':  
                pygame.draw.rect(surface, 'green', draw_rect)
            elif self.value == 'T': 
                pygame.draw.rect(surface, 'red', draw_rect)
            elif isinstance(self.value, int):
                pygame.draw.rect(surface, 'lavender', draw_rect)
            else:
                pygame.draw.rect(surface, 'white', draw_rect)
        
      
        pygame.draw.rect(surface, 'black', draw_rect, 1)
        
        if self.revealed and isinstance(self.value, int) and self.value != 0:
            text = normal_font.render(str(self.value), False, 'black')
            text_rect = text.get_rect(center=draw_rect.center)
            surface.blit(text, text_rect)

    def is_clicked(self, mouse_pos, offset_x, offset_y):
        adjusted_rect = pygame.Rect(
            self.rect.x + offset_x,
            self.rect.y + offset_y,
            self.rect.width,
            self.rect.height
        )
        return adjusted_rect.collidepoint(mouse_pos)


class Game:
    def __init__(self):
        
        self.state = "intro"
        self.cell_size = 32
        self.cells = []
        self.level = 1
        self.screen_width = 500
        self.screen_height = 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Gem Hunter")
        
        self.hint_sound = pygame.mixer.Sound('Source code/assets/sounds/hint.wav')
        self.win_sound = pygame.mixer.Sound('Source code/assets/sounds/win.mp3')
        self.lose_sound = pygame.mixer.Sound('Source code/assets/sounds/lose.mp3')
        self.bgm = pygame.mixer.Sound('Source code/assets/sounds/bgm.mp3')
        
        self.total_gems = 0
        self.revealed_gems = 0

    def create_menu_buttons(self):
        self.start_button = Button(self.screen_width//2 - 100, 250, 200, 60, "Start Game")
        self.quit_button = Button(self.screen_width//2 - 100, 350, 200, 60, "Quit")
        
    def generate_level(self):
        if self.level >= 3:
            algorithm = random.choice([solve_cnf_pysat, solve_cnf_backtracking])
        else:
            algorithm = random.choice([solve_cnf_pysat, solve_cnf_backtracking, solve_cnf_brute_force])
        self.loading_message = f"Generating puzzle using {algorithm.__name__}..."
        if self.level == 4:
            self.cell_size = 16
        self.state = "loading"
        self.cells = []
        
        try:
            self.matrix = read_matrix(f"Source code/testcases/input/input_{self.level}.txt")
            cnf, variables = prepare_cnf_data(self.matrix)
            model = algorithm(cnf, variables)
            
            if model is None:
                self.state = "intro"
                self.loading_message = "Solver failed. Trying again..."
                return
                
            self.matrix = interpret_model(model, self.matrix)
            # self.cells = [[Cell(i, j, self.cell_size, self.matrix[i][j]) for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))]
            # self.cells = []
            for i in range(len(self.matrix)):
                row = []
                for j in range(len(self.matrix[0])):
                    value = self.matrix[i][j]
                    cell = Cell(i, j, self.cell_size, value)
                    if isinstance(value, int): 
                        cell.revealed = True
                    row.append(cell)
                self.cells.append(row)

            self.total_gems = sum(1 for row in self.cells for cell in row if cell.value == 'G')
            self.revealed_gems = 0
            
            self.state = "playing"
        except Exception as e:
            print(f"Error generating level: {e}")
            self.state = "intro"
            self.loading_message = f"Error: {e}"
    
    def menu_scene(self):
        self.screen.fill('black')
        
        title_text = header_font.render("Gem Hunter", True, 'white')
        title_rect = title_text.get_rect(center=(self.screen_width//2, 150))
        self.screen.blit(title_text, title_rect)
        
        self.create_menu_buttons()
        self.start_button.draw(self.screen)
        self.quit_button.draw(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.start_button.is_clicked(pygame.mouse.get_pos(), True):
                    self.level = 1
                    self.generate_level()
                if self.quit_button.is_clicked(pygame.mouse.get_pos(), True):
                    self.state = "quit"
            if event.type == pygame.MOUSEMOTION:
                self.start_button.check_hover(pygame.mouse.get_pos())
                self.quit_button.check_hover(pygame.mouse.get_pos())
    
    def playing_scene(self):
        self.screen.fill('black')
        
        level_text = normal_font.render(f"LEVEL: {self.level}", True, 'white')
        level_rect = level_text.get_rect(topleft=(50, 20))
        self.screen.blit(level_text, level_rect)
        
        gem_text = normal_font.render(f"GEMS: {self.revealed_gems}/{self.total_gems}", True, 'white')
        gem_rect = gem_text.get_rect(topright=(self.screen_width - 50, 20))
        self.screen.blit(gem_text, gem_rect)
        
        grid_width = len(self.cells[0]) * self.cell_size
        grid_height = len(self.cells) * self.cell_size
        offset_x = (self.screen_width - grid_width) // 2
        offset_y = (self.screen_height - grid_height) // 2 + 10 
        
        for row in self.cells:
            for cell in row:
                cell.draw(self.screen, offset_x, offset_y)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for row in self.cells:
                        for cell in row:
                            if cell.is_clicked(pygame.mouse.get_pos(), offset_x, offset_y):
                                if not cell.revealed:
                                    if cell.value == 'T':
                                        cell.revealed = True 
                                        self.lose_sound.play()
                                        self.state = "lose"
                                    elif cell.value == 'G':
                                        self.hint_sound.play()
                                        cell.revealed = True
                                        self.revealed_gems += 1
                                        if self.revealed_gems >= self.total_gems:
                                            self.win_sound.play()
                                            self.state = "win"
                                    else:
                                        self.hint_sound.play()
                                        cell.revealed = True
                else:
                    mouse_pos = pygame.mouse.get_pos()
                    for row in self.cells:
                        for cell in row:
                            if cell.is_clicked(mouse_pos, offset_x, offset_y):
                                if not cell.revealed:
                                    cell.flagged = not cell.flagged
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.level += 1
                    self.generate_level()

    def win_scene(self):
        self.screen.fill('black')
        
        text = header_font.render("You Win!", True, 'white')
        text_rect = text.get_rect(center=(self.screen_width//2, self.screen_height//2 - 50))
        self.screen.blit(text, text_rect)
        
        level_text = normal_font.render(f"Completed Level {self.level}", True, 'white')
        level_rect = level_text.get_rect(center=(self.screen_width//2, self.screen_height//2))
        self.screen.blit(level_text, level_rect)
        
        next_text = small_font.render("Press any key to continue", True, 'white')
        next_rect = next_text.get_rect(center=(self.screen_width//2, self.screen_height//2 + 50))
        self.screen.blit(next_text, next_rect)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = "quit"
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                self.level += 1
                self.generate_level()
    
    def lose_scene(self):
        self.screen.fill('black')
        
        text = header_font.render("You Lose!", True, 'white')
        text_rect = text.get_rect(center=(self.screen_width//2, self.screen_height//2 - 50))
        self.screen.blit(text, text_rect)
        
        level_text = normal_font.render(f"Failed Level {self.level}", True, 'white')
        level_rect = level_text.get_rect(center=(self.screen_width//2, self.screen_height//2))
        self.screen.blit(level_text, level_rect)
        
        next_text = small_font.render("Press any key to return to menu", True, 'white')
        next_rect = next_text.get_rect(center=(self.screen_width//2, self.screen_height//2 + 50))
        self.screen.blit(next_text, next_rect)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = "quit"
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                self.state = "intro"
    
    def loading_scene(self):
        self.screen.fill('black')
        
        text = normal_font.render(self.loading_message, True, 'white')
        text_rect = text.get_rect(center=(self.screen_width//2, self.screen_height//2))
        self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        self.generate_level()
    
    def quit_scene(self):
        pygame.quit()
    
    def run(self):
        clock = pygame.time.Clock()
        self.bgm.play(-1)
        while self.state != "quit":
            if self.state == "intro":
                if not pygame.mixer.get_busy(): 
                    self.bgm.play(-1)
                self.menu_scene()
            elif self.state == "playing":
                self.bgm.stop()
                self.playing_scene()
            elif self.state == "win":
                self.win_scene()
            elif self.state == "lose":
                self.lose_scene()
            elif self.state == "loading":
                self.loading_scene()
                
            pygame.display.flip()
            clock.tick(60) 
        
        self.quit_scene()

if __name__ == "__main__":
    game = Game()
    game.run()

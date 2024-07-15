import pygame
import ui
import sys

pygame.init()
pygame.display.set_caption("AI Demo")


window = pygame.display.set_mode((1000, 800), pygame.RESIZABLE)
clock = pygame.time.Clock()

base = ui.Game(window, window.get_rect())

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    window.fill((0, 0, 0))
    base.rect = window.get_rect()
    base.render()
    pygame.display.update()
    clock.tick(60)

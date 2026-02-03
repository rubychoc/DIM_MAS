"""
Decentralized Multi-Agent System for Coalition Formation
A classroom simulation where students autonomously form optimal working groups.
"""

import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from typing import List, Set, Tuple, Optional
import random


# Name generation for students
FIRST_NAMES = [
    'Alex', 'Blake', 'Casey', 'Dana', 'Eli', 'Finn', 'Gray', 'Harper',
    'Ian', 'Jordan', 'Kelly', 'Logan', 'Morgan', 'Noah', 'Olivia', 'Parker',
    'Quinn', 'Riley', 'Sam', 'Taylor', 'Uma', 'Val', 'West', 'Xander',
    'Yara', 'Zoe', 'Aria', 'Beau', 'Cleo', 'Drew', 'Ellis', 'Finley',
    'Gale', 'Hunter', 'Indie', 'Jules', 'Kai', 'Lane', 'Max', 'Nova',
    'Opal', 'Peyton', 'Quincy', 'Remy', 'Sage', 'Tate', 'Unity', 'Vic',
    'Winter', 'Yarn', 'Zen', 'Azure'
]

def generate_name(id: int) -> str:
    """Generate a name for a student based on their ID."""
    return FIRST_NAMES[id % len(FIRST_NAMES)]


class Student:
    """Represents a student with skills, personality, and preferences."""
    
    def __init__(self, id: int, skills: np.ndarray, friends: Set[int], 
                 is_leader: bool, name: str):
        self.id = id
        self.name = name
        self.skills = skills  # Vector of competencies
        self.friends = friends  # Set of friend IDs
        self.is_leader = is_leader  # Leader vs Follower


class Entity(Agent):
    """
    An Entity is either a single student or a merged group.
    It acts as the negotiator for all its members.
    """
    
    def __init__(self, unique_id: int, model: Model, members: List[Student], x: int = 0, y: int = 0):
        super().__init__(model)
        self.unique_id = unique_id
        self.members = members  # List of Student objects
        self.size = len(members)
        self.state = "searching"  # searching, negotiating, locked
        self.pending_offer = None  # Stores incoming offer
        self.offer_utility = 0  # Expected utility from pending offer
        self.x = x  # X position in grid
        self.y = y  # Y position in grid
        self.last_action = None  # Track what this entity is doing
        
    def get_member_ids(self) -> Set[int]:
        """Returns set of all student IDs in this entity."""
        return {s.id for s in self.members}
    
    def get_display_name(self) -> str:
        """Returns display name for this entity."""
        if self.size == 1:
            return self.members[0].name
        else:
            # Find the group number (if entity still exists in model)
            multi_member_entities = [e for e in self.model.entities if e.size > 1]
            try:
                group_num = multi_member_entities.index(self) + 1
                return f"Group {group_num}"
            except ValueError:
                # Entity was removed from model (merged into another)
                # Return a generic name based on member names
                member_names = [m.name for m in self.members]
                if len(member_names) <= 2:
                    return " + ".join(member_names)
                else:
                    return f"{member_names[0]} + {len(member_names)-1} others"
    
    def get_combined_skills(self) -> np.ndarray:
        """Returns average skills of the group."""
        return np.mean([s.skills for s in self.members], axis=0)
    
    def get_leader_ratio(self) -> float:
        """Returns ratio of leaders in the group."""
        if self.size == 0:
            return 0
        return sum(s.is_leader for s in self.members) / self.size
    
    def calculate_skill_complementarity(self, student: Student, coalition: List[Student]) -> float:
        """
        Calculate skill complementarity for a student with coalition members.
        Higher score when skills are diverse.
        """
        comp_score = 0
        for other in coalition:
            if other.id != student.id:
                # Use negative correlation to reward diversity
                skill_diff = np.abs(student.skills - other.skills).mean()
                comp_score += skill_diff
        return comp_score
    
    def calculate_social_satisfaction(self, student: Student, coalition: List[Student]) -> float:
        """Count how many friends are in the coalition."""
        coalition_ids = {s.id for s in coalition if s.id != student.id}
        return len(student.friends.intersection(coalition_ids))
    
    def calculate_role_balance(self, student: Student, coalition: List[Student]) -> float:
        """
        Calculate personality balance score.
        Optimal: 0 < leader_ratio <= 0.4
        """
        leader_count = sum(s.is_leader for s in coalition)
        total = len(coalition)
        leader_ratio = leader_count / total if total > 0 else 0
        
        if leader_ratio == 0:
            return -0.5  # No leader penalty
        elif 0 < leader_ratio <= 0.4:
            return 1.0  # Optimal balance
        else:
            return -0.2  # Too many leaders penalty
    
    def calculate_utility(self, student: Student, coalition: List[Student]) -> float:
        """Calculate individual utility for a student in a coalition."""
        comp = self.calculate_skill_complementarity(student, coalition)
        social = self.calculate_social_satisfaction(student, coalition)
        role = self.calculate_role_balance(student, coalition)
        
        return self.model.ws * comp + self.model.wf * social + self.model.wp * role
    
    def calculate_total_utility(self, other_entity: 'Entity') -> float:
        """Calculate total utility if merging with another entity."""
        merged_members = self.members + other_entity.members
        total_utility = 0
        
        for student in merged_members:
            total_utility += self.calculate_utility(student, merged_members)
        
        return total_utility
    
    def get_acceptance_threshold(self) -> float:
        """
        Dynamic threshold based on time remaining.
        Entities become less picky as time runs out.
        """
        progress = self.model.elapsed_time / self.model.total_time if self.model.total_time > 0 else 0
        # Start with high standards, lower them as time passes
        return 1.0 * (1 - progress * 0.7)  # Drops from 1.0 to 0.3
    
    def move(self):
        """Move randomly to discover other entities with adaptive movement."""
        if self.state == "locked":
            return
        
        # Calculate active entities count for adaptive movement
        current_entities = len([e for e in self.model.entities if e.state not in ["locked", "locked (backup)"]])
        initial_entities = self.model.n_students
        
        # When fewer entities exist, allow longer jumps to explore more efficiently
        density_factor = current_entities / initial_entities if initial_entities > 0 else 0
        
        # Base radius is 1, but can increase to 2 when very sparse
        move_radius = 1 if density_factor > 0.3 else 2
        
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=move_radius
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        
        # Update x, y properties to match new position
        self.x, self.y = new_position
    
    def discover_neighbors(self) -> List['Entity']:
        """Find other entities in immediate vicinity with dynamic search radius."""
        # Calculate dynamic radius based on entity density
        initial_entities = self.model.n_students  # Started with n students (one entity each)
        current_entities = len([e for e in self.model.entities if e.state not in ["locked", "locked (backup)"]])
        
        # As entities merge, increase search radius to compensate for lower density
        # Radius ranges from 1 (full density) to 3 (very sparse)
        if current_entities == 0:
            search_radius = 3
        else:
            density_factor = current_entities / initial_entities
            search_radius = max(1, min(3, int(2 / (density_factor + 0.1))))
        
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=search_radius
        )
        return [n for n in neighbors if isinstance(n, Entity) and n.state not in ["locked", "locked (backup)"]]
    
    def make_offer(self, other: 'Entity') -> bool:
        """
        Evaluate and potentially make an offer to merge with another entity.
        Returns True if offer was made.
        """
        # Check if merge is possible (size constraint)
        if self.size + other.size > self.model.k:
            self.last_action = {
                'type': 'skip',
                'reason': 'Would exceed group size',
                'target': other.unique_id
            }
            return False
        
        # Calculate expected utility
        merged_utility = self.calculate_total_utility(other)
        
        # Compare with current utility
        current_utility = sum(self.calculate_utility(s, self.members) for s in self.members)
        current_utility += sum(other.calculate_utility(s, other.members) for s in other.members)
        
        threshold = self.get_acceptance_threshold()
        
        if merged_utility >= current_utility * threshold:
            other.pending_offer = self
            other.offer_utility = merged_utility
            other.state = "negotiating"
            
            self.last_action = {
                'type': 'offer',
                'initiatorId': self.unique_id,
                'targetId': other.unique_id,
                'initiatorName': self.get_display_name(),
                'targetName': other.get_display_name(),
                'currentUtility': round(current_utility, 2),
                'mergedUtility': round(merged_utility, 2),
                'threshold': round(threshold, 2)
            }
            self.model.negotiations.append(self.last_action)
            
            return True
        
        self.last_action = {
            'type': 'reject-offer',
            'reason': 'Utility too low',
            'initiatorId': self.unique_id,
            'targetId': other.unique_id,
            'initiatorName': self.get_display_name(),
            'targetName': other.get_display_name(),
            'mergedUtility': round(merged_utility, 2),
            'threshold': round(current_utility * threshold, 2)
        }
        
        return False
    
    def evaluate_offer(self) -> bool:
        """
        Evaluate pending offer and decide whether to accept.
        Returns True if accepted.
        """
        if self.pending_offer is None:
            return False
        
        # Re-check size constraint before accepting (in case offerer merged with others)
        if self.size + self.pending_offer.size > self.model.k:
            self.last_action = {
                'type': 'reject',
                'rejectorId': self.unique_id,
                'fromId': self.pending_offer.unique_id,
                'rejectorName': self.get_display_name(),
                'fromName': self.pending_offer.get_display_name(),
                'reason': 'Would exceed group size (offerer size changed)'
            }
            self.model.negotiations.append(self.last_action)
            
            self.pending_offer = None
            self.offer_utility = 0
            self.state = "searching"
            return False
        
        threshold = self.get_acceptance_threshold()
        current_utility = sum(self.calculate_utility(s, self.members) for s in self.members)
        
        # Accept if offered utility is better than current state
        if self.offer_utility >= current_utility * threshold:
            self.last_action = {
                'type': 'accept',
                'acceptorId': self.unique_id,
                'fromId': self.pending_offer.unique_id,
                'acceptorName': self.get_display_name(),
                'fromName': self.pending_offer.get_display_name(),
                'offerUtility': round(self.offer_utility, 2),
                'threshold': round(current_utility * threshold, 2)
            }
            self.model.negotiations.append(self.last_action)
            return True
        
        # Reject
        self.last_action = {
            'type': 'reject',
            'rejectorId': self.unique_id,
            'fromId': self.pending_offer.unique_id,
            'rejectorName': self.get_display_name(),
            'fromName': self.pending_offer.get_display_name(),
            'reason': 'Below threshold'
        }
        self.model.negotiations.append(self.last_action)
        
        self.pending_offer = None
        self.offer_utility = 0
        self.state = "searching"
        return False
    
    def merge_with(self, other: 'Entity'):
        """Merge with another entity (binding commitment)."""
        # Safety check: prevent merging if it would exceed k
        if self.size + other.size > self.model.k:
            # This shouldn't happen, but if it does, reject the merge
            self.last_action = {
                'type': 'reject',
                'rejectorId': self.unique_id,
                'fromId': other.unique_id,
                'rejectorName': self.get_display_name(),
                'fromName': other.get_display_name(),
                'reason': f'Safety check: Would exceed k (sizes: {self.size} + {other.size} > {self.model.k})'
            }
            self.model.negotiations.append(self.last_action)
            
            # Reset the acceptor's state
            other.pending_offer = None
            other.offer_utility = 0
            other.state = "searching"
            return
        
        old_size = self.size
        merger_name = self.get_display_name()
        mergee_name = other.get_display_name()
        
        self.members.extend(other.members)
        self.size = len(self.members)
        
        self.last_action = {
            'type': 'merge',
            'mergerId': self.unique_id,
            'mergeeId': other.unique_id,
            'mergerName': merger_name,
            'mergeeName': mergee_name,
            'newSize': self.size,
            'oldSizes': [old_size, other.size]
        }
        self.model.negotiations.append(self.last_action)
        
        # Lock if reached target size
        if self.size >= self.model.k:
            self.state = "locked"
        else:
            self.state = "searching"
        
        # Remove the other entity from the simulation
        if other.pos is not None:
            self.model.grid.remove_agent(other)
        if other in self.model.entities:
            self.model.entities.remove(other)

    
    def step(self):
        """Execute one step of the entity's behavior."""
        if self.state == "locked" or self.state == "locked (backup)":
            return
        
        # Handle pending offer
        if self.state == "negotiating":
            if self.evaluate_offer():
                # Accept and merge
                offerer = self.pending_offer
                # Check if offerer still exists (it might have been merged into another entity)
                if offerer in self.model.entities:
                    offerer.merge_with(self)
                else:
                    # Offerer was removed, reset our state
                    self.pending_offer = None
                    self.offer_utility = 0
                    self.state = "searching"
            return
        
        # Move and discover
        self.move()
        neighbors = self.discover_neighbors()
        
        # Try to make an offer to a random neighbor
        if neighbors:
            target = self.random.choice(neighbors)
            if target.state == "searching":
                self.make_offer(target)


class ClassroomModel(Model):
    """The main simulation model."""
    
    def __init__(self, n_students: int, k: int, grid_size: int, 
                 total_time: float, n_skills: int = 3,
                 ws: float = 0.4, wf: float = 0.3, wp: float = 0.3):
        super().__init__()
        self.n_students = n_students
        self.k = k  # Target group size
        self.grid_size = grid_size
        self.total_time = total_time  # Total simulation time in seconds
        self.n_skills = n_skills
        self.elapsed_time = 0.0  # Current elapsed time in seconds
        self.time_delta = 0.5  # Each tick advances time by 0.5 seconds
        self.current_step = 0  # Internal tick counter (for history)
        self.running = False
        
        # Global preference weights (same for all students)
        self.ws = ws  # Weight for skill complementarity
        self.wf = wf  # Weight for social satisfaction
        self.wp = wp  # Weight for personality balance
        
        # Track negotiations and history
        self.negotiations = []  # Track current step negotiations
        self.history = []  # Store snapshots for back/forward
        self.history_index = 0  # Current position in history
        self.students = []  # Store all students
        
        # Create grid
        self.grid = MultiGrid(grid_size, grid_size, torus=True)
        
        # Track active entities and next available entity ID
        self.entities = []
        self.next_entity_id = 0  # Counter for unique entity IDs
        
        # Create students
        self._create_students()
        
        # Generate unique positions for all students
        available_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        self.random.shuffle(available_positions)
        
        if len(self.students) > len(available_positions):
            raise ValueError(f"Grid too small: {len(self.students)} students need {grid_size}x{grid_size}={len(available_positions)} cells")
        
        # Create initial entities (one per student)
        for i, student in enumerate(self.students):
            x, y = available_positions[i]
            entity = Entity(self.next_entity_id, self, [student], x, y)
            self.next_entity_id += 1
            self.entities.append(entity)
            
            # Place at unique position
            self.grid.place_agent(entity, (x, y))
        
        # Save initial snapshot
        self.save_snapshot()
        
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Welfare": self.compute_total_welfare,
                "Num_Entities": lambda m: len(m.entities),
                "Num_Locked": lambda m: sum(1 for e in m.entities if e.state == "locked")
            }
        )
    
    def _create_students(self):
        """Create student agents with random attributes."""
        self.students = []

        for i in range(self.n_students):
            # Random skills
            skills = np.random.rand(self.n_skills)
            
            # Random friends (2-5 friends)
            n_friends = random.randint(2, 5)
            friends = set(random.sample(range(self.n_students), 
                                       min(n_friends, self.n_students - 1)))
            friends.discard(i)  # Can't be friends with self
            
            # Random personality (30% leaders)
            is_leader = random.random() < 0.3
            
            # Generate name
            name = generate_name(i)
            
            self.students.append(Student(i, skills, friends, is_leader, name))
        
        # Make friendships bidirectional
        for i in range(len(self.students)):
            student = self.students[i]
            for friend_id in list(student.friends):  # Convert to list to avoid set modification during iteration
                if friend_id < len(self.students):
                    self.students[friend_id].friends.add(i)
    
    def print_student_roster(self):
        """Print detailed information about all students."""
        print("\n" + "="*80)
        print("STUDENT ROSTER - Initial Characteristics")
        print("="*80)
        print(f"\nGlobal Preference Weights (same for all students):")
        print(f"  - Skill Complementarity (w_s): {self.w_s:.2f}")
        print(f"  - Social Satisfaction (w_f): {self.w_f:.2f}")
        print(f"  - Personality Balance (w_p): {self.w_p:.2f}")
        print()
        
        # Collect all students from entities
        all_students = []
        for entity in self.entities:
            all_students.extend(entity.members)
        
        all_students.sort(key=lambda s: s.id)
        
        for student in all_students:
            print(f"\n  Student {student.id}:")
            print(f"    Personality: {'LEADER' if student.is_leader else 'Follower'}")
            print(f"    Skills: {student.skills}")
            print(f"    Friends: {sorted(list(student.friends)) if student.friends else 'None'}")
        
        print("\n" + "="*80 + "\n")
    
    def compute_total_welfare(self) -> float:
        """Calculate total social welfare across all entities."""
        total = 0
        for entity in self.entities:
            for student in entity.members:
                total += entity.calculate_utility(student, entity.members)
        return total
    
    def detect_deadlock(self) -> bool:
        """
        Detect if simulation is in deadlock state where no more merges are possible.
        Returns True if deadlock detected.
        """
        # Get all non-locked entities
        active_entities = [e for e in self.entities if e.state not in ["locked", "locked (backup)"]]
        
        if len(active_entities) <= 1:
            return False  # Can't deadlock with 0 or 1 entity
        
        # Check if any pair of active entities can merge
        for i, entity1 in enumerate(active_entities):
            for entity2 in active_entities[i+1:]:
                if entity1.size + entity2.size <= self.k:
                    return False  # At least one valid merge possible
        
        return True  # No valid merges possible - deadlock!
    
    def break_lowest_utility_groups(self, num_groups: int = 2):
        """
        Break up the specified number of groups with lowest utility.
        Returns list of newly created singleton entities.
        """
        # Get all non-locked entities
        active_entities = [e for e in self.entities if e.state not in ["locked", "locked (backup)"]]
        
        if len(active_entities) < num_groups:
            return []  # Not enough entities to break up
        
        # Calculate utility for each entity
        entity_utilities = []
        for entity in active_entities:
            if entity.size > 1:  # Only consider groups, not singletons
                total_utility = sum(entity.calculate_utility(student, entity.members) 
                                  for student in entity.members)
                entity_utilities.append((entity, total_utility))
        
        if len(entity_utilities) < num_groups:
            return []  # Not enough multi-member groups to break up
        
        # Sort by utility (lowest first)
        entity_utilities.sort(key=lambda x: x[1])
        
        # Break up the lowest utility groups
        new_entities = []
        for i in range(num_groups):
            entity_to_break, utility = entity_utilities[i]
            
            # Log the breakup
            self.negotiations.append({
                'type': 'deadlock-breakup',
                'entityId': entity_to_break.unique_id,
                'entityName': entity_to_break.get_display_name(),
                'size': entity_to_break.size,
                'utility': round(utility, 2),
                'reason': 'Deadlock detected - breaking up low utility group'
            })
            
            # Remove entity from grid and list
            if entity_to_break.pos is not None:
                self.grid.remove_agent(entity_to_break)
            if entity_to_break in self.entities:
                self.entities.remove(entity_to_break)
            
            # Create singleton entities for each member
            for student in entity_to_break.members:
                # Try to place near original position
                x = entity_to_break.x if entity_to_break.x < self.grid_size else self.random.randrange(self.grid_size)
                y = entity_to_break.y if entity_to_break.y < self.grid_size else self.random.randrange(self.grid_size)
                
                new_entity = Entity(self.next_entity_id, self, [student], x, y)
                self.next_entity_id += 1
                new_entity.state = "searching"
                new_entities.append(new_entity)
                self.entities.append(new_entity)
                self.grid.place_agent(new_entity, (x, y))
        
        return new_entities
    
    def save_snapshot(self):
        """Save current state for history navigation."""
        snapshot = {
            'step': self.current_step,
            'elapsed_time': self.elapsed_time,
            'entities': [{
                'id': e.unique_id,
                'members': [m.id for m in e.members],
                'state': e.state,
                'x': e.x,
                'y': e.y,
                'size': e.size
            } for e in self.entities],
            'negotiations': self.negotiations.copy()
        }
        
        # If we've gone back in history and then do something new,
        # truncate the future history
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(snapshot)
        self.history_index = len(self.history) - 1
    
    def restore_snapshot(self, index: int) -> bool:
        """Restore state from a snapshot."""
        if index < 0 or index >= len(self.history):
            return False
        
        snapshot = self.history[index]
        self.current_step = snapshot['step']
        self.elapsed_time = snapshot['elapsed_time']
        self.negotiations = snapshot['negotiations'].copy()
        self.history_index = index
        
        # Rebuild entities from snapshot
        # First, remove all current entities from grid
        for entity in self.entities:
            if entity.pos is not None:
                self.grid.remove_agent(entity)
        
        self.entities = []
        max_id = -1
        for entity_data in snapshot['entities']:
            # Find the student objects for this entity's members
            members = [self.students[member_id] for member_id in entity_data['members']]
            
            entity = Entity(
                entity_data['id'],
                self,
                members,
                entity_data['x'],
                entity_data['y']
            )
            
            # Restore entity state
            entity.state = entity_data['state']
            entity.size = entity_data['size']
            
            self.entities.append(entity)
            self.grid.place_agent(entity, (entity.x, entity.y))
            
            # Track max ID to update next_entity_id
            max_id = max(max_id, entity_data['id'])
        
        # Update next_entity_id to be one more than the highest ID in the snapshot
        self.next_entity_id = max_id + 1
        
        return True
    
    def step_forward(self) -> bool:
        """Move forward one step in history."""
        if self.history_index < len(self.history) - 1:
            return self.restore_snapshot(self.history_index + 1)
        return False
    
    def step_back(self) -> bool:
        """Move back one step in history."""
        if self.history_index > 0:
            return self.restore_snapshot(self.history_index - 1)
        return False
    
    def handle_orphans(self):
        """At end of simulation, handle orphans by forming optimally-sized groups."""
        orphans = [e for e in self.entities if e.size < self.k]
        
        if not orphans:
            return
        
        # Collect all orphaned students
        all_orphan_students = []
        for orphan in orphans:
            all_orphan_students.extend(orphan.members)
        
        # Remove orphans from entities list
        self.entities = [e for e in self.entities if e.size >= self.k]
        
        total_orphans = len(all_orphan_students)
        
        # Determine optimal number of groups
        num_groups = max(1, round(total_orphans / self.k))
        
        # Distribute students as evenly as possible
        base_size = total_orphans // num_groups
        extra = total_orphans % num_groups
        
        start_idx = 0
        for i in range(num_groups):
            # Some groups get one extra student
            group_size = base_size + (1 if i < extra else 0)
            group_members = all_orphan_students[start_idx:start_idx + group_size]
            
            # Create new entity for this backup group
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            backup_entity = Entity(self.next_entity_id, self, group_members, x, y)
            self.next_entity_id += 1
            backup_entity.state = "locked (backup)"
            
            self.entities.append(backup_entity)
            self.grid.place_agent(backup_entity, (x, y))
            
            start_idx += group_size
    
    def step(self):
        """Execute one step of the simulation."""
        self.negotiations = []  # Clear negotiations for this step
        
        # Shuffle and activate all entities
        entity_list = list(self.entities)
        self.random.shuffle(entity_list)
        for entity in entity_list:
            if entity in self.entities:  # Check if still active after potential merges
                entity.step()
        
        # Check for deadlock after all entities have acted
        if self.detect_deadlock():
            # Break up the two lowest utility groups
            self.break_lowest_utility_groups(num_groups=2)
        
        # Validation: Verify all students are accounted for
        total_students_in_entities = sum(e.size for e in self.entities)
        if total_students_in_entities != self.n_students:
            print(f"WARNING: Student count mismatch! Expected {self.n_students}, found {total_students_in_entities}")
            print(f"Entities: {[(e.unique_id, e.size, [s.id for s in e.members]) for e in self.entities]}")
        
        # Check if all groups are locked (simulation complete)
        all_locked = all(e.state in ["locked", "locked (backup)"] for e in self.entities)
        if all_locked and len(self.entities) > 0:
            self.running = False
        
        # Advance time
        self.elapsed_time += self.time_delta
        self.current_step += 1
        self.save_snapshot()
        
        # Collect data
        if hasattr(self, 'datacollector'):
            self.datacollector.collect(self)
        
        # Check if time is up
        if self.elapsed_time >= self.total_time:
            self.handle_orphans()
            self.running = False


def print_detailed_results(model: ClassroomModel):
    """Print comprehensive results with detailed group information."""
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    print(f"\nSimulation completed in {model.current_step} steps")
    print(f"Target group size (k): {model.k}")
    print(f"Total students: {model.n_students}")
    
    # Count students in groups
    total_students_in_groups = sum(e.size for e in model.entities)
    
    print(f"\nFinal number of groups (entities): {len(model.entities)}")
    print(f"  - Locked groups (reached size k): {sum(1 for e in model.entities if e.state == 'locked')}")
    print(f"  - Still searching: {sum(1 for e in model.entities if e.state != 'locked')}")
    print(f"\nStudents accounted for: {total_students_in_groups}/{model.n_students}")
    
    if total_students_in_groups != model.n_students:
        print(f"  ⚠️  WARNING: {model.n_students - total_students_in_groups} students are missing!")
    
    print(f"\nNote: An 'entity' can be a single student or a merged group.")
    print(f"      Entities become 'locked' when they reach the target size of {model.k}.")
    print(f"\nTotal Social Welfare: {model.compute_total_welfare():.2f}")
    
    print("\n" + "-"*80)
    print("DETAILED GROUP BREAKDOWN")
    print("-"*80)
    
    # Track which students are in which groups
    all_assigned_students = set()
    
    for i, entity in enumerate(model.entities, 1):
        leader_count = sum(s.is_leader for s in entity.members)
        follower_count = entity.size - leader_count
        leader_ratio = entity.get_leader_ratio()
        
        # Track students
        for s in entity.members:
            all_assigned_students.add(s.id)
        
        print(f"\n{'█'*60}")
        print(f"GROUP {i} - {entity.state.upper()} - Size: {entity.size}/{model.k}")
        print(f"{'█'*60}")
        print(f"Composition: {leader_count} Leader(s), {follower_count} Follower(s) ({leader_ratio:.1%} leaders)")
        
        # Calculate group-level metrics
        avg_skills = entity.get_combined_skills()
        group_utility = sum(entity.calculate_utility(s, entity.members) for s in entity.members)
        
        print(f"Average Skills: {avg_skills}")
        print(f"Total Group Utility: {group_utility:.2f}")
        print(f"\nMembers:")
        
        for student in entity.members:
            # Calculate individual utility
            ind_utility = entity.calculate_utility(student, entity.members)
            
            # Count friends in group
            friends_in_group = entity.get_member_ids().intersection(student.friends)
            friends_in_group.discard(student.id)
            
            print(f"\n  • Student {student.id} ({'LEADER' if student.is_leader else 'Follower'})")
            print(f"      Skills: {student.skills}")
            print(f"      Individual Utility: {ind_utility:.2f}")
            print(f"      Friends in group: {sorted(list(friends_in_group)) if friends_in_group else 'None'}")
    
    # Check for missing students
    all_student_ids = set(range(model.n_students))
    missing_students = all_student_ids - all_assigned_students
    
    if missing_students:
        print(f"\n{'='*80}")
        print(f"⚠️  MISSING STUDENTS: {sorted(list(missing_students))}")
        print(f"{'='*80}")
    
    print("\n" + "="*80 + "\n")


def run_simulation(n_students: int = 20, k: int = 4, grid_size: int = 10, 
                   max_steps: int = 100, seed: Optional[int] = 45):
    """
    Run the coalition formation simulation.
    
    Parameters:
    - n_students: Number of students in the classroom
    - k: Target group size
    - grid_size: Size of the grid (grid_size x grid_size)
    - max_steps: Maximum number of simulation steps
    - seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    model = ClassroomModel(n_students, k, grid_size, max_steps)
    
    # Print initial student roster
    model.print_student_roster()
    
    print("Starting simulation...\n")
    
    # Run simulation
    for _ in range(max_steps):
        model.step()
        if not model.running:
            break
    
    # Print detailed results
    print_detailed_results(model)
    
    # Get data for analysis
    data = model.datacollector.get_model_vars_dataframe()
    
    return model, data


if __name__ == "__main__":
    # Example run
    model, data = run_simulation(
        n_students=20,
        k=4,
        grid_size=10,
        max_steps=1000,
        seed=42
    )
    
    # Display welfare progression
    print("\n=== Welfare Progression (every 20 steps) ===")
    for i in range(0, len(data), 20):
        print(f"Step {i:3d}: Welfare={data.iloc[i]['Total_Welfare']:7.2f}, "
              f"Entities={int(data.iloc[i]['Num_Entities']):3d}, "
              f"Locked={int(data.iloc[i]['Num_Locked']):3d}")

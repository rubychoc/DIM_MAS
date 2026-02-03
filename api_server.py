"""
FastAPI server for Coalition Formation Simulator
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from coalition_formation import ClassroomModel

app = FastAPI(title="Coalition Formation API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active simulations in memory
simulations: Dict[str, ClassroomModel] = {}


class SimulationConfig(BaseModel):
    n_students: int
    k: int
    grid_size: int
    total_time: float
    ws: float
    wf: float
    wp: float


class SimulationResponse(BaseModel):
    simulation_id: str
    elapsed_time: float
    running: bool


class StateResponse(BaseModel):
    elapsed_time: float
    total_time: float
    running: bool
    entities: List[Dict]
    negotiations: List[Dict]
    total_welfare: float
    num_entities: int
    num_locked: int


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse("index.html")


@app.post("/simulation/create", response_model=SimulationResponse)
async def create_simulation(config: SimulationConfig):
    """Create a new simulation with the given configuration."""
    try:
        # Create simulation ID
        sim_id = str(uuid.uuid4())
        
        # Create model
        model = ClassroomModel(
            n_students=config.n_students,
            k=config.k,
            grid_size=config.grid_size,
            total_time=config.total_time,
            ws=config.ws,
            wf=config.wf,
            wp=config.wp
        )
        model.running = True
        
        # Store simulation
        simulations[sim_id] = model
        
        return SimulationResponse(
            simulation_id=sim_id,
            elapsed_time=model.elapsed_time,
            running=model.running
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/{sim_id}/step", response_model=StateResponse)
async def step_simulation(sim_id: str):
    """Execute one step of the simulation."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    model = simulations[sim_id]
    
    if model.running:
        model.step()
    
    return get_simulation_state(model)


@app.get("/simulation/{sim_id}/state", response_model=StateResponse)
async def get_state(sim_id: str):
    """Get current state of the simulation."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    model = simulations[sim_id]
    return get_simulation_state(model)


@app.post("/simulation/{sim_id}/step-back", response_model=StateResponse)
async def step_back(sim_id: str):
    """Step back one step in history."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    model = simulations[sim_id]
    model.step_back()
    
    return get_simulation_state(model)


@app.post("/simulation/{sim_id}/step-forward", response_model=StateResponse)
async def step_forward(sim_id: str):
    """Step forward one step in history."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    model = simulations[sim_id]
    model.step_forward()
    
    return get_simulation_state(model)


@app.delete("/simulation/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a simulation."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulations[sim_id]
    return {"message": "Simulation deleted"}


@app.get("/simulation/{sim_id}/students")
async def get_students(sim_id: str):
    """Get all students in the simulation."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    model = simulations[sim_id]
    
    students = []
    for student in model.students:
        students.append({
            'id': student.id,
            'name': student.name,
            'skills': student.skills.tolist(),
            'friends': list(student.friends),
            'isLeader': student.is_leader
        })
    
    return students


@app.get("/simulation/{sim_id}/final-groups")
async def get_final_groups(sim_id: str):
    """Get final groups with utility calculations."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    model = simulations[sim_id]
    
    groups = []
    for i, entity in enumerate(model.entities):
        # Calculate group total utility
        group_utility = sum(entity.calculate_utility(student, entity.members) for student in entity.members)
        
        # Calculate individual utilities for each member
        members_with_utility = []
        for m in entity.members:
            individual_utility = entity.calculate_utility(m, entity.members)
            members_with_utility.append({
                'id': m.id, 
                'name': m.name, 
                'isLeader': m.is_leader, 
                'skills': m.skills.tolist(), 
                'friends': list(m.friends),
                'utility': round(individual_utility, 2)
            })
        
        display_name = entity.members[0].name if entity.size == 1 else f"Group {i + 1}"
        
        groups.append({
            'id': entity.unique_id,
            'displayName': display_name,
            'members': members_with_utility,
            'size': entity.size,
            'state': entity.state,
            'groupUtility': round(group_utility, 2)
        })
    
    return groups


def get_simulation_state(model: ClassroomModel) -> StateResponse:
    """Helper function to convert model state to response."""
    entities_data = []
    for entity in model.entities:
        entities_data.append({
            'id': entity.unique_id,
            'members': [{'id': m.id, 'name': m.name, 'isLeader': m.is_leader, 'skills': m.skills.tolist(), 'friends': list(m.friends)} for m in entity.members],
            'size': entity.size,
            'state': entity.state,
            'x': entity.x,
            'y': entity.y
        })
    
    num_locked = sum(1 for e in model.entities if e.state in ['locked', 'locked (backup)'])
    
    return StateResponse(
        elapsed_time=model.elapsed_time,
        total_time=model.total_time,
        running=model.running,
        entities=entities_data,
        negotiations=model.negotiations,
        total_welfare=model.compute_total_welfare(),
        num_entities=len(model.entities),
        num_locked=num_locked
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

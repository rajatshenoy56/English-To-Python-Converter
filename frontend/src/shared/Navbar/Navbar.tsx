import React, { Fragment } from "react";
import {
  Button,
  Form,
  FormControl,
  Nav,
  Navbar,
  NavDropdown,
} from "react-bootstrap";
import {NavLink} from 'react-router-dom'

const CustomNavbar = (props: any) => {
  return (
    <Fragment>
    <Navbar bg="success" expand="lg">
      <Navbar.Brand href="/">NL 2 Code</Navbar.Brand>
      <Navbar.Toggle aria-controls="basic-navbar-nav" />
      <Navbar.Collapse id="basic-navbar-nav">
        <Nav className="mr-auto">
          <Nav.Link href="/textinput">Text Input</Nav.Link>
          <Nav.Link href="/fileinput">File Input</Nav.Link>
          <Nav.Link href="/audioinput">Audio Input</Nav.Link>
        </Nav>
      </Navbar.Collapse>
    </Navbar>
    </Fragment>
	
  );
}

export default CustomNavbar;
